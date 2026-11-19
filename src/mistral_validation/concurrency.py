"""Exercise twenty authenticated requests against the protected external route."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.client import HTTPSConnection, HTTPException
import json
import socket
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
from typing import Any

from .serving import MODEL, SERVED_MODEL
from .validation import PAYLOAD, valid_chat, validate


REQUEST_COUNT = 20
TIMEOUT_SECONDS = 120


def chat_outcome(endpoint, token, index, gate):
    parsed = urlsplit(endpoint)
    connection = HTTPSConnection(parsed.hostname, parsed.port, timeout=TIMEOUT_SECONDS)
    gate.wait()
    started = time.monotonic()
    deadline = started + TIMEOUT_SECONDS
    status = None
    result = "transport_failed"
    timer = None
    connected = threading.Event()
    connect_errors = []

    def connect():
        try:
            connection.connect()
        except (OSError, HTTPException, ValueError):
            connect_errors.append(True)
        finally:
            if time.monotonic() >= deadline:
                connection.close()
            connected.set()

    try:
        # DNS resolution does not honor socket timeouts. A daemon connector
        # bounds the caller's wait even when the system resolver stalls.
        threading.Thread(target=connect, daemon=True).start()
        if not connected.wait(max(0, deadline - time.monotonic())):
            raise TimeoutError()
        if time.monotonic() >= deadline:
            raise TimeoutError()
        if connect_errors:
            raise OSError()
        connection.sock.settimeout(max(0.001, deadline - time.monotonic()))
        sock = connection.sock

        def interrupt():
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        timer = threading.Timer(max(0.001, deadline - time.monotonic()), interrupt)
        timer.daemon = True
        timer.start()
        connection.request("POST", parsed.path, body=PAYLOAD,
                           headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
        connection.sock.settimeout(max(0.001, deadline - time.monotonic()))
        response = connection.getresponse()
        status = response.status
        body = bytearray()
        while len(body) <= 1_000_000 and not response.isclosed():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            # HTTPResponse can detach a closing connection's socket from the
            # connection object, so retain it before reading the response.
            sock.settimeout(remaining)
            chunk = response.read1(min(65536, 1_000_001 - len(body)))
            if not chunk:
                break
            body.extend(chunk)
        if status in (401, 403):
            result = "authentication_failed"
        elif status != 200:
            result = "http_failed"
        else:
            result = "passed" if valid_chat(body) else "invalid_chat"
    except TimeoutError:
        result = "timeout"
    except (OSError, HTTPException, ValueError):
        result = "transport_failed"
    finally:
        if timer is not None:
            timer.cancel()
        connection.close()
    elapsed = time.monotonic() - started
    if elapsed > TIMEOUT_SECONDS:
        result = "timeout"
    return {"request_id": index, "http_status": status, "result": result,
            "elapsed_seconds": round(elapsed, 3)}


def check(oc, cfg, route_url, token, report_dir, oc_error):
    namespace = cfg["namespace"]
    evidence: dict[str, Any] = {"model": SERVED_MODEL, "effective_gpu_count": None,
                "request_count": REQUEST_COUNT, "timeout_seconds": TIMEOUT_SECONDS,
                "elapsed_seconds": 0.0, "requests": [], "result": "prerequisite_failed"}

    def finish(detail):
        timestamp = datetime.now(timezone.utc)
        evidence["failure_detail"] = detail
        try:
            root = Path(report_dir)
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"concurrency-{namespace}-{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
            path.write_text(json.dumps({"timestamp_utc": timestamp.isoformat(), "namespace": namespace,
                                        **evidence}, indent=2) + "\n")
        except OSError:
            print("FAIL concurrency: cannot save report; check REPORT_DIR")
            return 1
        passed = evidence["result"] == "passed"
        print(f"{'PASS' if passed else 'FAIL'} concurrency: {detail}; report {path}")
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
        url = route_url or service["status"]["url"]
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or (parsed.port is not None and parsed.port < 1)):
            raise ValueError()
        endpoint = url.rstrip("/") + "/v1/chat/completions"
    except (oc_error, KeyError, TypeError, ValueError):
        return finish("cannot establish deployed GPU allocation or route; inspect InferenceService")
    gate = threading.Barrier(REQUEST_COUNT)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=REQUEST_COUNT) as pool:
        futures = [pool.submit(chat_outcome, endpoint, token, index, gate)
                   for index in range(1, REQUEST_COUNT + 1)]
        evidence["requests"] = [future.result() for future in futures]
    evidence["elapsed_seconds"] = round(time.monotonic() - started, 3)
    passed = all(item["result"] == "passed" for item in evidence["requests"])
    evidence["result"] = "passed" if passed else "failed"
    return finish("20/20 valid responses" if passed else
                  "one or more requests failed; inspect per-request outcomes and model readiness, access, and logs")
