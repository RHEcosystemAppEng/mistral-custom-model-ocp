"""Exercise preparation identity guards through the operator's make targets."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_preparation import FAKE_OC, IMAGES, ROOT


class PreparationIdentityWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = self.root / "cluster.json"
        self.mutations = self.root / "mutations.jsonl"
        layout = self.root / "layout.json"
        layout.write_text(json.dumps({
            "model": "mistral-large-instruct-2411", "source_images": IMAGES, "assembly": "overlay",
            "files": [{"image": part, "source": f"/weights/file{part}", "target": f"file{part}",
                       "size": 1, "sha256": "a" * 64} for part in range(1, 5)]}))
        self.env = {**os.environ, "OC": str(FAKE_OC), "LAYOUT": str(layout),
                    "NAMESPACE": "mistral-custom-model-ocp", "PVC_NAME": "mistral-weights",
                    "GPU_COUNT": "4", "PREPARE_TIMEOUT": "1s", "SERVING_TIMEOUT": "15s",
                    "REPORT_DIR": str(self.root / "reports"),
                    "FAKE_OC_WEIGHT_IMAGES": json.dumps(IMAGES),
                    "FAKE_OC_RUNTIME_IMAGE": "cp.icr.io/cp/cpd/mistral-inference-server@sha256:03512d44a97adb6aff115e025df819ec05b249728471effe6a5615880af593a9",
                    "FAKE_OC_PREPARATION_STATE": str(self.state),
                    "FAKE_OC_MARKER": str(self.root / "serving"),
                    "FAKE_OC_MUTATIONS": str(self.mutations)}

    def run_make(self, target, **settings):
        scenario = "preparation_success" if target == "prepare" else "serving_success"
        return subprocess.run(["make", "-s", "--no-print-directory", target], cwd=ROOT,
                              env={**self.env, "FAKE_OC_SCENARIO": scenario, **settings},
                              capture_output=True, text=True)

    def prepare_claim(self, **settings):
        result = self.run_make("prepare", **settings)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.mutations.unlink()
        return json.loads(self.state.read_text())

    def assert_install_rejected_without_changes(self, **settings):
        result = self.run_make("install", **settings)
        self.assertNotEqual(result.returncode, 0)
        # With no mutating oc commands, existing serving resources cannot change.
        self.assertFalse(self.mutations.exists(), result.stdout + result.stderr)
        self.assertIn("PVC", result.stdout)
        self.assertIn("UID", result.stdout)
        self.assertIn("make prepare PVC_NAME=mistral-weights", result.stdout)

    def test_preparing_new_claim_authorizes_install_for_that_same_claim(self):
        state = self.prepare_claim(FAKE_OC_SCENARIO_PVC="no_pvc")
        self.assertEqual(state["pvc"]["metadata"]["uid"], "uid-mistral-weights")
        result = self.run_make("install")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(next((self.root / "reports").glob("serving-*.json")).read_text())
        self.assertTrue(report["inference_valid"])

    def test_retry_after_observer_disconnect_preserves_job_and_allows_install(self):
        interrupted = self.run_make("prepare", FAKE_OC_OBSERVER_INTERRUPTED="1")
        self.assertNotEqual(interrupted.returncode, 0)
        self.assertFalse(list((self.root / "reports").glob("preparation-*.json")))
        original = self.state.read_text()
        self.mutations.unlink()
        retried = self.run_make("prepare")
        self.assertEqual(retried.returncode, 0, retried.stdout + retried.stderr)
        self.assertFalse(self.mutations.exists())
        self.assertEqual(self.state.read_text(), original)
        evidence = json.loads(next((self.root / "reports").glob("preparation-*.json")).read_text())
        self.assertEqual(evidence["pvc"]["uid"], "uid-mistral-weights")
        self.assertEqual(evidence["verified_files"], 4)
        installed = self.run_make("install")
        self.assertEqual(installed.returncode, 0, installed.stdout + installed.stderr)

    def test_retry_recovers_after_local_report_write_failure(self):
        failed = self.run_make("prepare", FAKE_OC_BLOCK_REPORT_WRITE="1")
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("local report could not be saved", failed.stdout)
        original = self.state.read_text()
        (self.root / "reports").unlink()
        self.mutations.unlink()
        retried = self.run_make("prepare")
        self.assertEqual(retried.returncode, 0, retried.stdout + retried.stderr)
        self.assertFalse(self.mutations.exists())
        self.assertEqual(self.state.read_text(), original)
        evidence = json.loads(next((self.root / "reports").glob("preparation-*.json")).read_text())
        self.assertEqual(evidence["job"], "mistral-2411-prepare")
        self.assertEqual(evidence["verified_files"], 4)
        self.assertEqual(evidence["source_images"][0], {
            "reference": IMAGES[0], "observed_image_id": "observed-image-1"})

    def test_retry_does_not_reuse_mismatched_or_unverified_evidence(self):
        original = self.prepare_claim()
        for failure in ("wrong UID", "incomplete Job", "wrong image", "missing image ID", "unverified log", "wrong count", "changed inventory"):
            with self.subTest(failure=failure):
                state = deepcopy(original)
                settings = {}
                if failure == "wrong UID":
                    state["pvc"] = {"metadata": {"name": "mistral-weights", "uid": "replacement-uid"},
                                    "spec": {"accessModes": ["ReadWriteOnce"],
                                             "resources": {"requests": {"storage": "140Gi"}}}}
                elif failure == "incomplete Job":
                    state["job"]["status"] = {"conditions": [{"type": "Complete", "status": "False"}]}
                elif failure == "wrong image":
                    state["job"]["spec"]["template"]["spec"]["initContainers"][0]["image"] = "wrong-image"
                elif failure == "missing image ID":
                    state["pod"] = deepcopy(state["job"]["spec"]["template"])
                    state["pod"]["status"] = {"initContainerStatuses": []}
                elif failure == "wrong count":
                    settings["FAKE_OC_VERIFICATION_LOG"] = "PREPARATION_COMPLETE verified_files=1 peak_pvc_used_kib=1024"
                elif failure == "changed inventory":
                    prior_layout = json.loads(state["configmap"]["data"]["layout.json"])
                    prior_layout["files"][0]["sha256"] = "b" * 64
                    state["configmap"]["data"]["layout.json"] = json.dumps(prior_layout)
                else:
                    settings["FAKE_OC_VERIFICATION_LOG"] = "verification did not complete"
                self.state.write_text(json.dumps(state))
                self.mutations.unlink(missing_ok=True)
                retried = self.run_make("prepare", **settings)
                self.assertNotEqual(retried.returncode, 0, retried.stdout + retried.stderr)
                self.assertNotIn("reusing verified preparation", retried.stdout)
                self.assertIn("MODEL_ALREADY_COMPLETE", retried.stdout)
                self.assertTrue(self.mutations.exists())

    def test_install_rejects_a_recreated_claim_without_changing_serving_resources(self):
        state = self.prepare_claim()
        state["pvc"] = {"metadata": {"name": "mistral-weights", "uid": "replacement-uid"}}
        self.state.write_text(json.dumps(state))
        self.assert_install_rejected_without_changes()

    def test_install_rejects_unavailable_claim_without_changing_serving_resources(self):
        self.prepare_claim()
        self.assert_install_rejected_without_changes(FAKE_OC_SCENARIO_PVC="no_pvc")

    def test_install_rejects_incomplete_or_mismatched_identity_evidence(self):
        original = self.prepare_claim()
        template_path = ("job", "spec", "template")
        cases = [
            ("missing current UID", ("pvc",), {"metadata": {"name": "mistral-weights"}}),
            ("empty current UID", ("pvc",), {"metadata": {"name": "mistral-weights", "uid": ""}}),
            ("missing current name", ("pvc",), {"metadata": {"uid": "uid-mistral-weights"}}),
            ("unavailable current metadata", ("pvc",), {"metadata": None}),
            ("deleting claim", ("pvc",), {"metadata": {"name": "mistral-weights",
                "uid": "uid-mistral-weights", "deletionTimestamp": "2026-11-18T00:00:00Z"}}),
            ("legacy Job without UID evidence", (*template_path, "metadata"), {}),
            ("unavailable prepared UID", (*template_path, "metadata", "annotations"), None),
            ("empty prepared UID", (*template_path, "metadata", "annotations"),
                {"mistral-validation/pvc-uid": ""}),
            ("missing Job claim", (*template_path, "spec", "volumes"), []),
            ("wrong Job claim", (*template_path, "spec", "volumes"),
                [{"name": "model", "persistentVolumeClaim": {"claimName": "other-weights"}}]),
        ]
        for label, path, value in cases:
            with self.subTest(evidence=label):
                state = deepcopy(original)
                parent = state
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value
                self.state.write_text(json.dumps(state))
                self.assert_install_rejected_without_changes()

    def test_install_checks_the_actual_preparation_pod_identity(self):
        original = self.prepare_claim()
        for field in ("metadata", "volumes"):
            with self.subTest(field=field):
                state = deepcopy(original)
                pod = deepcopy(state["job"]["spec"]["template"])
                pod["status"] = {"initContainerStatuses": [{"imageID": f"observed-image-{i}"}
                                                          for i in range(1, 5)]}
                if field == "metadata":
                    pod["metadata"]["annotations"]["mistral-validation/pvc-uid"] = "different-uid"
                else:
                    pod["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"] = "other-weights"
                state["pod"] = pod
                self.state.write_text(json.dumps(state))
                self.assert_install_rejected_without_changes()

    def test_install_still_requires_verified_inventory_and_pinned_images(self):
        original = self.prepare_claim()
        for failure in ("incomplete Job", "wrong image", "missing image ID", "missing log", "unverified log"):
            with self.subTest(failure=failure):
                state = deepcopy(original)
                settings = {}
                if failure == "incomplete Job":
                    state["job"]["status"] = {"conditions": [{"type": "Complete", "status": "False"}]}
                elif failure == "wrong image":
                    state["job"]["spec"]["template"]["spec"]["initContainers"][0]["image"] = "wrong-image"
                elif failure == "missing image ID":
                    state["pod"] = deepcopy(state["job"]["spec"]["template"])
                    state["pod"]["status"] = {"initContainerStatuses": []}
                elif failure == "missing log":
                    settings["FAKE_OC_LOGS_UNAVAILABLE"] = "1"
                else:
                    settings["FAKE_OC_VERIFICATION_LOG"] = "verification did not complete"
                self.state.write_text(json.dumps(state))
                self.assert_install_rejected_without_changes(**settings)

    def test_prepare_requires_claim_identity_before_starting_a_job(self):
        self.state.write_text(json.dumps({"pvc": {"metadata": {"name": "mistral-weights"},
            "spec": {"storageClassName": "fast", "accessModes": ["ReadWriteOnce"],
                     "resources": {"requests": {"storage": "140Gi"}}}}}))
        result = self.run_make("prepare")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("UID", result.stdout)
        self.assertIn("make prepare", result.stdout)
        self.assertNotIn("job", json.loads(self.state.read_text()))
        self.assertFalse(list((self.root / "reports").glob("preparation-*.json")))

    def test_prepare_rejects_a_claim_replaced_during_preparation(self):
        result = self.run_make("prepare", FAKE_OC_RECREATE_PVC_AFTER_PREPARATION="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("UID changed", result.stdout)
        self.assertIn("make prepare", result.stdout)
        self.assertFalse(list((self.root / "reports").glob("preparation-*.json")))

    def test_prepare_requires_verification_logs_before_reporting_success(self):
        result = self.run_make("prepare", FAKE_OC_LOGS_UNAVAILABLE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("verification logs are unavailable", result.stdout)
        self.assertFalse(list((self.root / "reports").glob("preparation-*.json")))


if __name__ == "__main__":
    unittest.main()
