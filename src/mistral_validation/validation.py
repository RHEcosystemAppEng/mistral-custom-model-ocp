"""Validate the protected RHOAI external inference route."""

from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .serving import SERVED_MODEL


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


OPENER = build_opener(ProxyHandler({}), NoRedirect())
PAYLOAD = json.dumps({"model": SERVED_MODEL,
                      "messages": [{"role": "user", "content": "Reply briefly to confirm availability."}],
                      "max_tokens": 16, "temperature": 0}).encode()


def request_chat(url, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=PAYLOAD, headers=headers, method="POST")
    try:
        with OPENER.open(request, timeout=120) as response:
            return response.status, response.read(1_000_001)
    except HTTPError as error:
        error.close()
        return error.code, b""
    except (OSError, URLError, ValueError):
        return None, b""


def valid_chat(body):
    if len(body) > 1_000_000:
        return False
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return False
    if not isinstance(data, dict) or data.get("model") != SERVED_MODEL:
        return False
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False
    message = choices[0].get("message")
    return (isinstance(message, dict) and message.get("role") == "assistant"
            and isinstance(message.get("content"), str) and bool(message["content"].strip()))


def save_report(report_dir, namespace, evidence):
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    path = root / f"validation-{namespace}-{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    data = {"timestamp_utc": timestamp.isoformat(), "namespace": namespace, **evidence}
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def validate(oc, namespace, route_url, token, report_dir, oc_error):
    evidence = {"unauthenticated_status": None, "authenticated_status": None,
                "authenticated_chat_valid": False, "result": None}

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
        service = oc.get("inferenceservice", "mistral-large-instruct-2411", "-n", namespace)
    except oc_error:
        return finish("readiness_failed", "cannot inspect InferenceService; check cluster access and deployment")
    ready = any(item.get("type") == "Ready" and item.get("status") == "True"
                for item in service.get("status", {}).get("conditions", []))
    if not ready:
        return finish("readiness_failed", "InferenceService is not Ready; inspect predictor pods and events")
    annotations = service.get("metadata", {}).get("annotations", {})
    if annotations.get("security.opendatahub.io/enable-auth") != "true":
        return finish("authentication_failed", "InferenceService has not requested RHOAI platform authentication")
    url = route_url or service.get("status", {}).get("url", "")
    try:
        parsed = urlsplit(url)
    except ValueError:
        return finish("route_failed", "external route URL is invalid; set ROUTE_URL to its HTTPS base URL")
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        return finish("route_failed", "set ROUTE_URL to the external HTTPS route")
    endpoint = url.rstrip("/") + "/v1/chat/completions"
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
    if auth_status != 200 or not valid_chat(body):
        return finish("inference_failed", f"chat inference failed or response contract invalid (HTTP {auth_status}); inspect model logs")
    evidence["authenticated_chat_valid"] = True
    return finish("passed", "unauthenticated request rejected and authenticated chat response valid")
