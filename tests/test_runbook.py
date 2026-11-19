"""Exercise evidence and deletion guards through the operator CLI."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RunbookTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = self.root / "state.json"
        self.objects = {"pvc/mistral-weights": {
            "kind": "PersistentVolumeClaim", "metadata": {
                "name": "mistral-weights", "namespace": "runbook", "uid": "original"}}}
        self.state.write_text(json.dumps(self.objects))
        self.env = {**os.environ, "FAKE_OC_SCENARIO": "cleanup", "FAKE_OC_STATE": str(self.state)}
        self.env.pop("RHOAI_TOKEN", None)

    def invoke(self, command, *options, token=None):
        env = dict(self.env)
        if token:
            env["RHOAI_TOKEN"] = token
        return subprocess.run([
            sys.executable, "-m", "mistral_validation", command, "--namespace", "runbook",
            "--oc", str(ROOT / "tests/fake_oc.py"), "--report-dir", str(self.root), *options],
            env=env, capture_output=True, text=True)

    def test_missing_token_saves_incomplete_bundle_without_changing_cluster(self):
        result = self.invoke("verify-run")
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(next(self.root.glob("run-*/verification.json")).read_text())
        self.assertEqual(report["result"], "incomplete")
        self.assertEqual(report["human_review"]["conclusion"], "pending")
        self.assertTrue(all(item["result"] == "not_run" for item in report["checks"].values()))
        self.assertEqual(json.loads(self.state.read_text()), self.objects)
        self.assertTrue(next(self.root.glob("run-*/summary.md")).exists())

    def test_existing_claim_refused_and_token_never_persisted(self):
        result = self.invoke("verify-run", token="private-token-never-save")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(self.state.read_text()), self.objects)
        report = next(self.root.glob("run-*/verification.json")).read_text()
        self.assertNotIn("private-token", report + result.stdout + result.stderr)
        self.assertIn("fresh dedicated namespace", report)

    def test_purge_refuses_replacement_claim(self):
        report = self.root / "verification.json"
        report.write_text(json.dumps({"namespace": "runbook", "requested_configuration": {
            "pvc": {"name": "mistral-weights"}}, "tested_configuration": {"pvc_uid": "older-claim"}}))
        result = self.invoke("purge-run", "--run-report", str(report))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(self.state.read_text()), self.objects)

    def test_purge_refuses_replacement_service_when_claim_is_absent(self):
        objects = {"inferenceservice/mistral-large-instruct-2411": {
            "kind": "InferenceService", "metadata": {
                "name": "mistral-large-instruct-2411", "namespace": "runbook", "uid": "new-service"}}}
        self.state.write_text(json.dumps(objects))
        report = self.root / "verification.json"
        report.write_text(json.dumps({"namespace": "runbook", "requested_configuration": {
            "pvc": {"name": "mistral-weights"}}, "tested_configuration": {
                "pvc_uid": "old-claim", "inferenceservice_uid": "old-service"}}))
        result = self.invoke("purge-run", "--run-report", str(report))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(self.state.read_text()), objects)

    def test_cleanup_retry_checks_saved_volume_after_claim_is_gone(self):
        self.check_cleanup_retry(saved_volume=True)

    def test_successful_cleanup_preserves_failed_verification(self):
        self.check_cleanup_retry(saved_volume=False)

    def check_cleanup_retry(self, saved_volume):
        self.state.write_text("{}")
        report = self.root / "verification.json"
        report.write_text(json.dumps({"namespace": "runbook", "requested_configuration": {
            "pvc": {"name": "mistral-weights"}}, "result": "incomplete",
            "started_utc": "2026-11-18T00:00:00+00:00", "human_review": {"conclusion": "pending"},
            "checks": {"validate": {"result": "failed"}}}))
        if saved_volume:
            evidence = json.loads(report.read_text())
            evidence.update(backing_volume={"name": "old-pv", "uid": "old-pv-uid"}, storage_reclaim_policy="Delete")
            report.write_text(json.dumps(evidence))
        result = self.invoke("purge-run", "--run-report", str(report))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        evidence = json.loads(report.read_text())
        if saved_volume:
            self.assertTrue(evidence["backing_volume_removed"])
        self.assertEqual(evidence["result"], "incomplete")
        self.assertEqual(evidence["checks"]["validate"]["result"], "failed")
        self.assertEqual(evidence["checks"]["purge_repeat"]["result"], "passed")

    def test_purge_refuses_wrong_namespace(self):
        report = self.root / "verification.json"
        report.write_text(json.dumps({"namespace": "other", "requested_configuration": {
            "pvc": {"name": "mistral-weights"}}}))
        result = self.invoke("purge-run", "--run-report", str(report))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(self.state.read_text()), self.objects)


if __name__ == "__main__":
    unittest.main()
