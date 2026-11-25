"""Run chat inference and validate the protected RHOAI external route."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import NamedTuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .serving import SERVED_MODEL


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


OPENER = build_opener(ProxyHandler({}), NoRedirect())
PROMPT = "Reply briefly to confirm availability."
MISSING_TOKEN_MESSAGE = "FAIL authentication: set RHOAI_TOKEN to a RHOAI platform model access token"


def chat_payload(prompt: str, max_tokens: int) -> bytes:
    return json.dumps({"model": SERVED_MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()


PAYLOAD = chat_payload(PROMPT, 16)


def request_chat(url, token=None, *, payload=PAYLOAD):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=payload, headers=headers, method="POST")
    try:
        with OPENER.open(request, timeout=120) as response:
            return response.status, response.read(1_000_001)
    except HTTPError as error:
        error.close()
        return error.code, b""
    except (OSError, URLError, ValueError):
        return None, b""


class ChatReply(NamedTuple):
    content: str
    reached_limit: bool


def _chat_reply(body: bytes) -> ChatReply | None:
    if len(body) > 1_000_000:
        return None
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("model") != SERVED_MODEL:
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return ChatReply(content, choices[0].get("finish_reason") == "length")
    return None


def valid_chat(body):
    return _chat_reply(body) is not None


class RouteFailure(Exception):
    def __init__(self, result: str, action: str):
        super().__init__(action)
        self.result = result


def chat_endpoint(oc, namespace, route_url, oc_error):
    try:
        service = oc.get("inferenceservice", "mistral-large-instruct-2411", "-n", namespace)
    except oc_error as error:
        raise RouteFailure("readiness_failed", "cannot inspect InferenceService; check cluster access and deployment") from error
    ready = any(item.get("type") == "Ready" and item.get("status") == "True"
                for item in service.get("status", {}).get("conditions", []))
    if not ready:
        raise RouteFailure("readiness_failed", "InferenceService is not Ready; inspect predictor pods and events")
    annotations = service.get("metadata", {}).get("annotations", {})
    if annotations.get("security.opendatahub.io/enable-auth") != "true":
        raise RouteFailure("authentication_failed", "InferenceService has not requested RHOAI platform authentication")
    url = route_url or service.get("status", {}).get("url", "")
    try:
        parsed = urlsplit(url)
    except ValueError as error:
        raise RouteFailure("route_failed", "external route URL is invalid; set ROUTE_URL to its HTTPS base URL") from error
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise RouteFailure("route_failed", "set ROUTE_URL to the external HTTPS route")
    return url.rstrip("/") + "/v1/chat/completions"


def chat(oc, namespace, route_url, oc_error):
    prompt = os.environ.get("MISTRAL_CHAT_PROMPT")
    try:
        if prompt is None:
            prompt = "" if sys.stdin.isatty() else sys.stdin.buffer.read().decode("utf-8")
    except (OSError, UnicodeError):
        print("FAIL chat: cannot read a UTF-8 prompt from standard input", file=sys.stderr)
        return 1
    if not prompt.strip():
        print('Usage: make chat PROMPT="Explain why the sky is blue" or make chat < prompt.txt', file=sys.stderr)
        return 1
    try:
        max_tokens = int(os.environ.get("MAX_TOKENS", "256"))
        if max_tokens < 1:
            raise ValueError
    except ValueError:
        print("FAIL chat: MAX_TOKENS must be a positive integer", file=sys.stderr)
        return 1
    token = os.environ.get("RHOAI_TOKEN", "")
    if not token:
        print(MISSING_TOKEN_MESSAGE, file=sys.stderr)
        return 1
    try:
        endpoint = chat_endpoint(oc, namespace, route_url, oc_error)
    except RouteFailure as error:
        print(f"FAIL chat: {error}", file=sys.stderr)
        return 1
    status, body = request_chat(endpoint, token, payload=chat_payload(prompt, max_tokens))
    if status != 200:
        reason = f"route returned HTTP {status}" if status is not None else "cannot reach route; check DNS and TLS"
        print(f"FAIL chat: {reason}", file=sys.stderr)
        return 1
    reply = _chat_reply(body)
    if reply is None:
        print("FAIL chat: response has no usable assistant reply; inspect model logs", file=sys.stderr)
        return 1
    content = reply.content
    print(content, end="" if content.endswith("\n") else "\n")
    if reply.reached_limit:
        print(f"WARNING chat: reply reached MAX_TOKENS={max_tokens}", file=sys.stderr)
    return 0


def save_report(report_dir, namespace, evidence):
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    path = root / f"validation-{namespace}-{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    data = {"timestamp_utc": timestamp.isoformat(), "namespace": namespace, **evidence}
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def validate(oc, namespace, route_url, token, report_dir, oc_error, *, include_exchange=False):
    evidence: dict[str, int | bool | str | None] = {
        "unauthenticated_status": None, "authenticated_status": None,
        "authenticated_chat_valid": False, "result": None,
    }
    if include_exchange:
        evidence.update(prompt=None, response=None)

    def finish(result, action):
        evidence["result"] = result
        try:
            path = save_report(report_dir, namespace, evidence)
        except OSError:
            print("FAIL validation: cannot save report; check REPORT_DIR")
            return 1
        print(f"{'PASS' if result == 'passed' else 'FAIL'} validation: {action}; report {path}")
        return 0 if result == "passed" else 1

    try:
        endpoint = chat_endpoint(oc, namespace, route_url, oc_error)
    except RouteFailure as error:
        return finish(error.result, str(error))
    if include_exchange:
        evidence["prompt"] = PROMPT
    unauth_status, _ = request_chat(endpoint)
    evidence["unauthenticated_status"] = unauth_status
    if unauth_status is None:
        return finish("route_failed", "cannot connect to external route; check DNS, TLS, and route")
    if unauth_status == 503:
        return finish("readiness_failed", "route returned HTTP 503; inspect predictor readiness")
    if unauth_status in (404, 405):
        return finish("route_failed", f"external route returned HTTP {unauth_status}; check ROUTE_URL and path")
    if unauth_status not in (401, 403):
        return finish("authentication_failed", f"unauthenticated route request returned HTTP {unauth_status}; require platform token authentication")
    auth_status, body = request_chat(endpoint, token)
    evidence["authenticated_status"] = auth_status
    if auth_status is None:
        return finish("route_failed", "authenticated request could not reach route; check DNS and TLS")
    if auth_status in (401, 403):
        return finish("authentication_failed", f"platform token rejected with HTTP {auth_status}; check token and model access")
    if auth_status == 503:
        return finish("readiness_failed", "route returned HTTP 503; inspect predictor readiness")
    reply = _chat_reply(body) if auth_status == 200 else None
    if reply is None:
        return finish("inference_failed", f"chat inference failed or response contract invalid (HTTP {auth_status}); inspect model logs")
    evidence["authenticated_chat_valid"] = True
    if include_exchange:
        evidence["response"] = reply.content
    return finish("passed", "unauthenticated request rejected and authenticated chat response valid")
