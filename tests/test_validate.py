import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = Path(__file__).resolve().parents[1]
FAKE_OC = ROOT / "tests" / "fake_oc.py"
MODEL = "mistralai/mistral-large-instruct-2411"


class RouteHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.prompts.append(request["messages"][0]["content"])
        self.server.requests.append((self.path, self.headers.get("Authorization"), request["max_tokens"]))
        if self.server.route_status:
            status, payload = self.server.route_status, {"detail": "sensitive route error"}
        elif not self.headers.get("Authorization") and not self.server.allow_anonymous:
            status, payload = 401, {"detail": "sensitive denial"}
        elif self.headers.get("Authorization") and self.headers["Authorization"] != "Bearer platform-secret":
            status, payload = 403, {"detail": "sensitive denial"}
        else:
            status = 200
            payload = {"model": self.server.model, "choices": [{
                "message": {"role": "assistant", "content": self.server.content},
                "finish_reason": self.server.finish_reason,
            }]}
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ValidateWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        directory = Path(self.tmp.name)
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                        "-keyout", str(directory / "key.pem"), "-out", str(directory / "cert.pem"),
                        "-days", "1", "-subj", "/CN=localhost",
                        "-addext", "subjectAltName=DNS:localhost"], check=True, capture_output=True)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RouteHandler)
        self.server.model = MODEL
        self.server.content = "Sensitive generated answer"
        self.server.finish_reason = "stop"
        self.server.prompts = []
        self.server.requests = []
        self.server.allow_anonymous = False
        self.server.route_status = None
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(directory / "cert.pem", directory / "key.pem")
        self.server.socket = context.wrap_socket(self.server.socket, server_side=True)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"https://localhost:{self.server.server_port}"
        self.directory = directory

    def route_env(self, **extra):
        return {**os.environ, "OC": str(FAKE_OC), "FAKE_OC_SCENARIO": "validate_ready",
               "ROUTE_URL": self.url, "RHOAI_TOKEN": "platform-secret",
               "SSL_CERT_FILE": str(self.directory / "cert.pem"), "REPORT_DIR": str(self.directory), **extra}

    def run_validate(self, **extra):
        result = subprocess.run(["make", "-s", "--no-print-directory", "validate"],
                                cwd=ROOT, env=self.route_env(**extra), capture_output=True, text=True)
        reports = list(self.directory.glob("validation-*.json"))
        evidence = json.loads(reports[0].read_text()) if reports else None
        return result, evidence, reports[0].read_text() if reports else ""

    def run_chat(self, prompt, *variables, **extra):
        env = self.route_env(**extra)
        env.pop("PROMPT", None)
        env.pop("MAX_TOKENS", None)
        return subprocess.run(["make", "--no-print-directory", "chat", f"PYTHON={sys.executable}", *variables],
                              cwd=ROOT, env=env, input=prompt, capture_output=True, text=True, timeout=10)

    def test_chat_sends_one_authenticated_request_with_stdin_prompt(self):
        prompt = "  Explain this.\r\nInclude café.  \n"
        result = self.run_chat(prompt)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.prompts, [prompt])
        self.assertEqual(self.server.requests, [("/v1/chat/completions", "Bearer platform-secret", 256)])
        self.assertEqual(result.stdout, "Sensitive generated answer\n")
        self.assertEqual(result.stderr, "")
        self.assertEqual(list(self.directory.glob("*.json")), [])

    def test_chat_uses_literal_inline_prompt_instead_of_stdin(self):
        prompt = 'Explain "$5" and $(NAMESPACE).  '
        result = self.run_chat("Ignore this input", f"PROMPT={prompt}")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.prompts, [prompt])

    def test_chat_uses_active_oc_login_for_discovered_route(self):
        result = self.run_chat("Explain this", RHOAI_TOKEN="", ROUTE_URL="",
                               FAKE_OC_ROUTE_URL=self.url, FAKE_OC_TOKEN="platform-secret")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.requests, [("/v1/chat/completions", "Bearer platform-secret", 256)])
        self.assertEqual(result.stdout, "Sensitive generated answer\n")
        self.assertEqual(result.stderr, "")

    def test_chat_requires_explicit_token_for_other_route(self):
        result = self.run_chat("Explain this", RHOAI_TOKEN="", FAKE_OC_ROUTE_URL="https://model.example.test",
                               FAKE_OC_TOKEN="platform-secret")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("set RHOAI_TOKEN explicitly", result.stderr)
        self.assertEqual(self.server.requests, [])

    def test_chat_reports_missing_oc_login_for_discovered_route(self):
        result = self.run_chat("Explain this", RHOAI_TOKEN="", ROUTE_URL="",
                               FAKE_OC_ROUTE_URL=self.url, FAKE_OC_TOKEN="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("oc login", result.stderr)
        self.assertEqual(self.server.requests, [])

    def test_chat_prints_partial_reply_and_warns_when_token_limit_is_reached(self):
        self.server.content = "Partial reply"
        self.server.finish_reason = "length"
        result = self.run_chat("Tell me a story", "MAX_TOKENS=512")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.requests, [("/v1/chat/completions", "Bearer platform-secret", 512)])
        self.assertEqual(result.stdout, "Partial reply\n")
        self.assertIn("reached MAX_TOKENS=512", result.stderr)

    def test_chat_rejects_explicitly_blank_prompt_instead_of_reading_stdin(self):
        result = self.run_chat("Ignore this input", "PROMPT=  ")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn('make chat PROMPT="', result.stderr)
        self.assertIn("make chat < prompt.txt", result.stderr)
        self.assertEqual(self.server.requests, [])

    def test_chat_rejects_unready_service_even_with_an_explicit_route(self):
        result = self.run_chat("Explain this", FAKE_OC_SCENARIO="validate_unready")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("InferenceService is not Ready", result.stderr)
        self.assertEqual(self.server.requests, [])

    def test_chat_reports_request_failure_without_printing_the_error_body(self):
        self.server.route_status = 503
        result = self.run_chat("Explain this")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("HTTP 503", result.stderr)
        self.assertNotIn("sensitive route error", result.stderr)

    def test_chat_rejects_unusable_reply(self):
        self.server.content = "  \n"
        result = self.run_chat("Explain this")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("no usable assistant reply", result.stderr)

    def test_protected_route_accepts_platform_token_and_records_safe_evidence(self):
        self.server.content = "  Generated answer\nwith café.  "
        result, evidence, report = self.run_validate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(evidence["unauthenticated_status"], 401)
        self.assertTrue(evidence["authenticated_chat_valid"])
        self.assertEqual(evidence["result"], "passed")
        self.assertEqual(self.server.prompts, [evidence["prompt"], evidence["prompt"]])
        self.assertEqual(evidence["response"], self.server.content)
        self.assertNotIn("platform-secret", result.stdout + result.stderr + report)
        self.assertNotIn("Generated answer", result.stdout + result.stderr)
        self.assertNotIn(evidence["prompt"], result.stdout + result.stderr)
        self.assertNotIn("sensitive denial", result.stdout + result.stderr + report)

    def test_wrong_model_fails_inference_contract(self):
        self.server.model = "other-model"
        result, evidence, _ = self.run_validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "inference_failed")
        self.assertEqual(evidence["prompt"], self.server.prompts[0])
        self.assertIsNone(evidence["response"])
        self.assertIn("inference", result.stdout.lower())

    def test_missing_token_fails_without_sending_requests(self):
        result, evidence, _ = self.run_validate(RHOAI_TOKEN="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(evidence)
        self.assertIn("RHOAI_TOKEN", result.stdout + result.stderr)

    def test_unprotected_route_fails_authentication_check(self):
        self.server.allow_anonymous = True
        result, evidence, _ = self.run_validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "authentication_failed")
        self.assertIsNone(evidence["authenticated_status"])
        self.assertEqual(evidence["prompt"], self.server.prompts[0])
        self.assertIsNone(evidence["response"])

    def test_unready_service_fails_before_route_requests(self):
        result, evidence, _ = self.run_validate(FAKE_OC_SCENARIO="validate_unready")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "readiness_failed")
        self.assertIsNone(evidence["unauthenticated_status"])
        self.assertIsNone(evidence["prompt"])
        self.assertIsNone(evidence["response"])

    def test_service_without_platform_auth_fails_before_route_requests(self):
        result, evidence, _ = self.run_validate(FAKE_OC_SCENARIO="validate_no_platform_auth")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "authentication_failed")
        self.assertIsNone(evidence["unauthenticated_status"])

    def test_bad_route_reports_route_failure(self):
        self.server.route_status = 404
        result, evidence, report = self.run_validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "route_failed")
        self.assertNotIn("sensitive route error", result.stdout + result.stderr + report)

    def test_rejected_platform_token_reports_authentication_failure(self):
        result, evidence, report = self.run_validate(RHOAI_TOKEN="wrong-secret")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "authentication_failed")
        self.assertEqual(evidence["authenticated_status"], 403)
        self.assertNotIn("wrong-secret", result.stdout + result.stderr + report)


if __name__ == "__main__":
    unittest.main()
