"""Exercise near-limit inference through the operator CLI and a protected route."""

import json
import os
import subprocess
import unittest

import test_validate as route_tests
from test_validate import ROOT, FAKE_OC, RouteHandler


class ContextRouteHandler(RouteHandler):
    def do_POST(self):
        if self.path != "/tokenize" and int(self.headers["Content-Length"]) < 1000:
            return super().do_POST()
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.headers.get("Authorization") != "Bearer platform-secret":
            self.send_error(403)
            return
        # Fixture tokenizer: each whitespace-delimited word has two tokens;
        # the chat template, including generation marker, adds eleven.
        count = 2 * len(request["messages"][0]["content"].split()) + 11
        if self.path == "/tokenize":
            self.server.measured_messages.append(request["messages"])
            payload = {"count": count, "max_model_len": getattr(self.server, "model_length", 32768),
                       "tokens": [123] * count}
            if getattr(self.server, "bad_measurement", False):
                payload["count"] = "sensitive tokenization detail"
        else:
            self.server.context_requests.append(request)
            payload = {"model": self.server.model, "choices": [{"message": {
                "role": "assistant", "content": self.server.content}}],
                "usage": {"prompt_tokens": count, "completion_tokens": 3,
                          "total_tokens": count + 3}}
            if getattr(self.server, "truncated_input", False):
                payload["usage"] = {"prompt_tokens": 1024, "completion_tokens": 3, "total_tokens": 1027}
            if getattr(self.server, "bad_model", False):
                payload["model"] = "wrong-model"
            if getattr(self.server, "bad_usage", False):
                payload["usage"] = {"prompt_tokens": "sensitive usage detail"}
        status = getattr(self.server, "tokenize_status" if self.path == "/tokenize" else "context_status", 200)
        if status != 200:
            payload = {"detail": "sensitive context failure body"}
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.path != "/tokenize" and getattr(self.server, "disconnect", False):
            self.wfile.write(body[:10])
            self.close_connection = True
            return
        self.wfile.write(body)


class ContextWorkflowTests(unittest.TestCase):
    def setUp(self):
        route_tests.ValidateWorkflowTests.setUp(self)
        self.server.RequestHandlerClass = ContextRouteHandler
        self.server.measured_messages = []
        self.server.context_requests = []

    def run_check(self, **extra):
        env = {**os.environ, "OC": str(FAKE_OC), "FAKE_OC_SCENARIO": "validate_ready",
               "ROUTE_URL": self.url, "RHOAI_TOKEN": "platform-secret",
               "SSL_CERT_FILE": str(self.directory / "cert.pem"),
               "REPORT_DIR": str(self.directory), **extra}
        result = subprocess.run(["make", "-s", "--no-print-directory", "context-check"],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        reports = list(self.directory.glob("context-*.json"))
        raw = reports[0].read_text() if reports else ""
        return result, json.loads(raw) if raw else None, raw

    def test_measured_chat_nears_limit_and_records_safe_hardware_evidence(self):
        result, evidence, raw = self.run_check(GPU_COUNT="8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(evidence["result"], "passed")
        self.assertEqual(evidence["effective_gpu_count"], 4)
        self.assertEqual(evidence["configured_model_length"], 32768)
        self.assertEqual(evidence["reserved_output_tokens"], 16)
        self.assertGreaterEqual(evidence["input_tokens"] + 16, 32640)
        self.assertLessEqual(evidence["input_tokens"] + 16, 32768)
        self.assertIn("/tokenize", evidence["measurement_method"])
        self.assertEqual(evidence["completion_tokens"], 3)
        self.assertEqual(evidence["http_status"], 200)
        self.assertGreater(evidence["elapsed_seconds"], 0)
        self.assertEqual(len(self.server.context_requests), 1)
        sent = self.server.context_requests[0]
        self.assertIn(sent["messages"], self.server.measured_messages)
        self.assertEqual(sent["max_tokens"], 16)
        self.assertNotIn("truncate_prompt_tokens", sent)
        for secret in ("platform-secret", "Sensitive generated answer", self.url,
                       sent["messages"][0]["content"]):
            self.assertNotIn(secret, result.stdout + result.stderr + raw)
        self.assertFalse(list(self.directory.glob("concurrency-*.json")))

    def test_truncated_input_fails_and_reports_observed_token_counts(self):
        self.server.truncated_input = True
        result, evidence, raw = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "invalid_usage")
        self.assertEqual(evidence["prompt_tokens"], 1024)
        self.assertEqual(evidence["completion_tokens"], 3)
        self.assertEqual(evidence["total_tokens"], 1027)
        self.assertNotIn("Sensitive generated answer", raw + result.stdout + result.stderr)

    def test_disconnected_response_records_transport_failure(self):
        self.server.disconnect = True
        result, evidence, raw = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNotNone(evidence, result.stdout + result.stderr)
        self.assertEqual(evidence["result"], "transport_failed")
        self.assertGreater(evidence["input_tokens"], 32000)
        self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_unavailable_tokenizer_fails_before_near_limit_inference(self):
        self.server.tokenize_status = 404
        result, evidence, raw = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "tokenization_failed")
        self.assertEqual(evidence["tokenize_http_status"], 404)
        self.assertEqual(self.server.context_requests, [])
        self.assertNotIn("sensitive context failure body", raw + result.stdout + result.stderr)

    def test_different_server_model_length_fails_before_near_limit_inference(self):
        self.server.model_length = 16384
        result, evidence, _ = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "configuration_failed")
        self.assertEqual(self.server.context_requests, [])

    def test_malformed_token_count_does_not_leak_server_data(self):
        self.server.bad_measurement = True
        result, evidence, raw = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "tokenization_failed")
        self.assertEqual(self.server.context_requests, [])
        self.assertNotIn("sensitive tokenization detail", raw + result.stdout + result.stderr)

    def test_near_limit_rejection_has_its_own_http_outcome(self):
        self.server.context_status = 400
        result, evidence, raw = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "request_rejected")
        self.assertEqual(evidence["http_status"], 400)
        self.assertGreater(evidence["elapsed_seconds"], 0)
        self.assertNotIn("sensitive context failure body", raw + result.stdout + result.stderr)

    def test_near_limit_wrong_model_fails_response_contract(self):
        self.server.bad_model = True
        result, evidence, _ = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "invalid_chat")

    def test_malformed_usage_does_not_leak_server_data(self):
        self.server.bad_usage = True
        result, evidence, raw = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "invalid_usage")
        self.assertNotIn("sensitive usage detail", raw + result.stdout + result.stderr)

    def test_unprotected_route_stops_before_tokenization(self):
        self.server.allow_anonymous = True
        result, evidence, _ = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "prerequisite_failed")
        self.assertEqual(self.server.measured_messages, [])
        self.assertEqual(self.server.context_requests, [])

    def test_missing_token_stops_before_tokenization(self):
        result, evidence, _ = self.run_check(RHOAI_TOKEN="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RHOAI_TOKEN", result.stdout + result.stderr)
        self.assertIsNone(evidence)
        self.assertEqual(self.server.measured_messages, [])
