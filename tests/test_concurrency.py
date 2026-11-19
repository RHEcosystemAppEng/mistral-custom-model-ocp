"""Exercise the concurrency command through a protected HTTPS route."""

import json
import os
import subprocess
import threading
import unittest
from unittest.mock import patch
import time

import test_validate as route_tests
from test_validate import RouteHandler, ROOT, FAKE_OC


class ConcurrentRouteHandler(RouteHandler):
    def do_POST(self):
        if self.headers.get("Authorization"):
            with self.server.count_lock:
                self.server.auth_count += 1
                index = self.server.auth_count
            # The first authenticated request is the route precheck.
            if index > 1:
                try:
                    self.server.batch.wait(timeout=10)
                except threading.BrokenBarrierError:
                    self.send_error(500)
                    return
            if index == 2 and getattr(self.server, "batch_failure", False):
                self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps({"model": "wrong-model", "choices": [{"message": {
                    "role": "assistant", "content": "sensitive batch content"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        super().do_POST()


class ConcurrencyWorkflowTests(unittest.TestCase):
    # Reuse the HTTPS fixture without inheriting its route-validation tests.
    def setUp(self):
        route_tests.ValidateWorkflowTests.setUp(self)
        self.server.RequestHandlerClass = ConcurrentRouteHandler
        self.server.socket.listen(64)
        self.server.auth_count = 0
        self.server.count_lock = threading.Lock()
        self.server.batch = threading.Barrier(20)

    def run_check(self, **extra):
        env = {**os.environ, "OC": str(FAKE_OC), "FAKE_OC_SCENARIO": "validate_ready",
               "ROUTE_URL": self.url, "RHOAI_TOKEN": "platform-secret",
               "SSL_CERT_FILE": str(self.directory / "cert.pem"), "REPORT_DIR": str(self.directory), **extra}
        result = subprocess.run(["make", "-s", "--no-print-directory", "concurrency-check"],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        reports = list(self.directory.glob("concurrency-*.json"))
        raw = reports[0].read_text() if reports else ""
        return result, json.loads(raw) if raw else None, raw

    def test_twenty_requests_overlap_and_report_safe_results(self):
        result, evidence, raw = self.run_check(GPU_COUNT="8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(evidence["effective_gpu_count"], 4)
        self.assertEqual(evidence["request_count"], 20)
        self.assertEqual(evidence["timeout_seconds"], 120)
        self.assertEqual(evidence["result"], "passed")
        self.assertEqual(len(evidence["requests"]), 20)
        self.assertTrue(all(r["result"] == "passed" and r["http_status"] == 200 for r in evidence["requests"]))
        self.assertGreater(evidence["elapsed_seconds"], 0)
        for secret in ("platform-secret", "Sensitive generated answer", self.url):
            self.assertNotIn(secret, result.stdout + result.stderr + raw)

    def test_single_bad_response_fails_batch_without_losing_other_outcomes(self):
        self.server.batch_failure = True
        result, evidence, raw = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "failed")
        self.assertEqual(len(evidence["requests"]), 20)
        self.assertEqual(sum(r["result"] == "invalid_chat" for r in evidence["requests"]), 1)
        self.assertEqual(sum(r["result"] == "passed" for r in evidence["requests"]), 19)
        self.assertNotIn("sensitive batch content", raw + result.stdout + result.stderr)

    def test_unprotected_route_stops_before_batch(self):
        self.server.allow_anonymous = True
        result, evidence, _ = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "prerequisite_failed")
        self.assertEqual(evidence["requests"], [])
        self.assertEqual(self.server.auth_count, 0)


class RequestDeadlineTests(unittest.TestCase):
    def test_stalled_dns_returns_timeout_without_waiting_for_resolver(self):
        from mistral_validation import concurrency
        release = threading.Event()
        self.addCleanup(release.set)

        def stalled_dns(*args, **kwargs):
            release.wait(5)
            raise OSError("sensitive resolver detail")

        with patch("socket.getaddrinfo", side_effect=stalled_dns), \
                patch.object(concurrency, "TIMEOUT_SECONDS", 0.05):
            started = time.monotonic()
            outcome = concurrency.chat_outcome("https://example.invalid/v1/chat/completions",
                                               "secret", 1, threading.Barrier(1))
            elapsed = time.monotonic() - started
        self.assertEqual(outcome["result"], "timeout")
        self.assertIsNone(outcome["http_status"])
        self.assertLess(elapsed, 1)
        self.assertNotIn("secret", json.dumps(outcome))
