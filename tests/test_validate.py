import json
import os
from pathlib import Path
import ssl
import subprocess
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
        if self.server.route_status:
            status, payload = self.server.route_status, {"detail": "sensitive route error"}
        elif not self.headers.get("Authorization") and not self.server.allow_anonymous:
            status, payload = 401, {"detail": "sensitive denial"}
        elif self.headers.get("Authorization") and self.headers["Authorization"] != "Bearer platform-secret":
            status, payload = 403, {"detail": "sensitive denial"}
        else:
            status = 200
            payload = {"model": self.server.model, "choices": [{"message": {"role": "assistant", "content": self.server.content}}]}
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
        self.server.prompts = []
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

    def run_validate(self, **extra):
        env = {**os.environ, "OC": str(FAKE_OC), "FAKE_OC_SCENARIO": "validate_ready",
               "ROUTE_URL": self.url, "RHOAI_TOKEN": "platform-secret",
               "SSL_CERT_FILE": str(self.directory / "cert.pem"), "REPORT_DIR": str(self.directory), **extra}
        result = subprocess.run(["make", "-s", "--no-print-directory", "validate"],
                                cwd=ROOT, env=env, capture_output=True, text=True)
        reports = list(self.directory.glob("validation-*.json"))
        evidence = json.loads(reports[0].read_text()) if reports else None
        return result, evidence, reports[0].read_text() if reports else ""

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
