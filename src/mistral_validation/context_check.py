"""Measure a near-32K chat request and verify inference on the protected route."""

from datetime import datetime, timezone
from http.client import HTTPException
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

from .serving import MODEL, SERVED_MODEL
from .validation import OPENER, valid_chat, validate


MODEL_LENGTH = 32768
OUTPUT_TOKENS = 16
MAX_HEADROOM = 128
TIMEOUT_SECONDS = 120
MEASUREMENT_METHOD = "vLLM POST /tokenize with chat template and generation prompt"


class CheckFailure(Exception):
    pass


def post(endpoint, payload, token):
    request = Request(endpoint, data=json.dumps(payload).encode(), method="POST",
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    try:
        with OPENER.open(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read(1_000_001)
            expected = response.headers.get("Content-Length")
            if expected is not None and len(body) < min(int(expected), 1_000_001):
                raise CheckFailure("transport_failed")
            return response.status, body
    except HTTPError as error:
        error.close()
        return error.code, b""
    except TimeoutError:
        raise CheckFailure("timeout") from None
    except URLError as error:
        raise CheckFailure("timeout" if isinstance(error.reason, TimeoutError) else "transport_failed") from None
    except (OSError, HTTPException, ValueError):
        raise CheckFailure("transport_failed") from None


def decode(body):
    try:
        data = json.loads(body) if len(body) <= 1_000_000 else None
    except (ValueError, UnicodeDecodeError):
        data = None
    if not isinstance(data, dict):
        raise CheckFailure("invalid_response")
    return data


def measure_request(base, token, evidence):
    # Search synthetic padding length; only the final measured request is sent
    # for generation. Tokenization includes the server's chat-template overhead.
    low, high = 0, MODEL_LENGTH
    while low <= high:
        repeats = (low + high) // 2
        payload = {"model": SERVED_MODEL, "add_generation_prompt": True,
                   "add_special_tokens": False,
                   "messages": [{"role": "user", "content":
                                 "Read the padding and reply OK." + " padding" * repeats + "\nReply OK."}]}
        status, body = post(base + "/tokenize", payload, token)
        evidence["tokenize_http_status"] = status
        if status != 200:
            raise CheckFailure("authentication_failed" if status in (401, 403) else "tokenization_failed")
        measured = decode(body)
        length, count = measured.get("max_model_len"), measured.get("count")
        if type(length) is not int or length != MODEL_LENGTH:
            raise CheckFailure("configuration_failed")
        evidence["observed_model_length"] = length
        if type(count) is not int or count < 1:
            raise CheckFailure("tokenization_failed")
        total = count + OUTPUT_TOKENS
        if MODEL_LENGTH - MAX_HEADROOM <= total <= MODEL_LENGTH:
            evidence["input_tokens"] = count
            evidence["input_plus_reserved_tokens"] = total
            payload.update(max_tokens=OUTPUT_TOKENS, temperature=0)
            return payload
        if total > MODEL_LENGTH:
            high = repeats - 1
        else:
            low = repeats + 1
    raise CheckFailure("cannot_construct_near_limit")


def check(oc, cfg, route_url, token, report_dir, oc_error):
    namespace = cfg["namespace"]
    evidence: dict[str, Any] = {
        "model": SERVED_MODEL, "effective_gpu_count": None,
        "configured_model_length": MODEL_LENGTH, "observed_model_length": None,
        "measurement_method": MEASUREMENT_METHOD, "reserved_output_tokens": OUTPUT_TOKENS,
        "max_headroom_tokens": MAX_HEADROOM, "input_tokens": None,
        "input_plus_reserved_tokens": None, "prompt_tokens": None,
        "completion_tokens": None, "total_tokens": None, "http_status": None,
        "tokenize_http_status": None, "elapsed_seconds": 0.0,
        "timeout_seconds": TIMEOUT_SECONDS, "result": "prerequisite_failed"}

    def finish(detail):
        timestamp = datetime.now(timezone.utc)
        evidence["failure_detail"] = detail
        try:
            root = Path(report_dir)
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"context-{namespace}-{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
            path.write_text(json.dumps({"timestamp_utc": timestamp.isoformat(), "namespace": namespace,
                                        **evidence}, indent=2) + "\n")
        except OSError:
            print("FAIL context: cannot save report; check REPORT_DIR")
            return 1
        passed = evidence["result"] == "passed"
        print(f"{'PASS' if passed else 'FAIL'} context: result={evidence['result']}; {detail}; "
              f"input_tokens={evidence['input_tokens']} reserved_output_tokens={OUTPUT_TOKENS}; "
              f"method={MEASUREMENT_METHOD}; report {path}")
        return 0 if passed else 1

    if validate(oc, namespace, route_url, token, report_dir, oc_error):
        return finish("protected route precheck failed; inspect validation report")
    try:
        service = oc.get("inferenceservice", MODEL, "-n", namespace)
        resources = service["spec"]["predictor"]["model"]["resources"]
        count = int(resources["limits"]["nvidia.com/gpu"])
        if count < 1 or int(resources.get("requests", {}).get("nvidia.com/gpu", count)) != count:
            raise ValueError()
        evidence["effective_gpu_count"] = count
        base = route_url or service["status"]["url"]
        parsed = urlsplit(base)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or (parsed.port is not None and parsed.port < 1)):
            raise ValueError()
    except (oc_error, KeyError, TypeError, ValueError):
        return finish("cannot establish deployed GPU allocation or route; inspect InferenceService")
    try:
        payload = measure_request(base.rstrip("/"), token, evidence)
    except CheckFailure as error:
        evidence["result"] = str(error)
        return finish("token measurement failed; inspect tokenizer endpoint, model length, access, and route")
    started = time.monotonic()
    try:
        status, body = post(base.rstrip("/") + "/v1/chat/completions", payload, token)
        evidence["http_status"] = status
        if status in (401, 403):
            raise CheckFailure("authentication_failed")
        if status in (400, 413, 422):
            raise CheckFailure("request_rejected")
        if status != 200:
            raise CheckFailure("http_failed")
        if not valid_chat(body):
            raise CheckFailure("invalid_chat")
        usage = decode(body).get("usage")
        if not isinstance(usage, dict):
            raise CheckFailure("invalid_usage")
        prompt, completion, total = (usage.get(key) for key in
                                     ("prompt_tokens", "completion_tokens", "total_tokens"))
        if (type(prompt) is not int or type(completion) is not int or type(total) is not int
                or min(prompt, completion, total) < 0):
            raise CheckFailure("invalid_usage")
        evidence.update(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
        if (prompt != evidence["input_tokens"] or not 1 <= completion <= OUTPUT_TOKENS
                or total != prompt + completion or total > MODEL_LENGTH):
            raise CheckFailure("invalid_usage")
        evidence["result"] = "passed"
    except CheckFailure as error:
        evidence["result"] = str(error)
    finally:
        evidence["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return finish("valid near-limit response" if evidence["result"] == "passed" else
                  "near-limit inference failed; inspect result, HTTP status, model capacity, and logs")
