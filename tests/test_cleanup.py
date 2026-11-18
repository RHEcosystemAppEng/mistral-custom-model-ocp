import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FAKE_OC = ROOT / "tests" / "fake_oc.py"
NAMESPACE = "cleanup-validation"
CLAIM = "custom-weights"
MODEL = "mistral-large-instruct-2411"


def resource(kind, name, owner=None, **fields):
    metadata = {"name": name, "namespace": NAMESPACE, "uid": f"uid-{kind}-{name}"}
    if owner:
        owner_kind, owner_name = owner
        metadata["ownerReferences"] = [{"kind": owner_kind, "name": owner_name,
                                         "uid": f"uid-{owner_kind}-{owner_name}"}]
    return {"kind": kind, "metadata": metadata, **fields}


class CleanupWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.state_path = Path(directory.name) / "cluster.json"
        self.env = {**os.environ, "OC": str(FAKE_OC), "NAMESPACE": NAMESPACE,
                    "PVC_NAME": CLAIM, "FAKE_OC_SCENARIO": "cleanup",
                    "FAKE_OC_STATE": str(self.state_path), "CLEANUP_TIMEOUT": "5s"}
        objects = {
            f"inferenceservice/{MODEL}": resource("InferenceService", MODEL),
            f"deployment/{MODEL}-predictor": resource("Deployment", f"{MODEL}-predictor", ("InferenceService", MODEL)),
            f"service/{MODEL}-predictor": resource("Service", f"{MODEL}-predictor", ("InferenceService", MODEL)),
            f"route/{MODEL}": resource("Route", MODEL, ("InferenceService", MODEL)),
            "servingruntime/mistral-runtime": resource("ServingRuntime", "mistral-runtime"),
            "configmap/mistral-2411-preparation": resource("ConfigMap", "mistral-2411-preparation"),
            "job/mistral-2411-prepare": resource("Job", "mistral-2411-prepare"),
            f"pvc/{CLAIM}": resource("PersistentVolumeClaim", CLAIM, status={"phase": "Bound"}),
            "secret/ibm-entitlement": resource("Secret", "ibm-entitlement"),
            "configmap/openshift-service-ca.crt": resource("ConfigMap", "openshift-service-ca.crt"),
            "deployment/unrelated": resource("Deployment", "unrelated"),
            "pod/predictor": resource("Pod", "predictor", ("Deployment", f"{MODEL}-predictor"),
                                      spec={"volumes": [{"persistentVolumeClaim": {"claimName": CLAIM}}]}),
            "pod/prepare": resource("Pod", "prepare", ("Job", "mistral-2411-prepare"),
                                    spec={"volumes": [{"persistentVolumeClaim": {"claimName": CLAIM}}]}),
        }
        for part in range(1, 5):
            name = f"mistral-2411-discover-{part}"
            objects[f"job/{name}"] = resource("Job", name)
            objects[f"pod/discover-{part}"] = resource("Pod", f"discover-{part}", ("Job", name))
        self.state_path.write_text(json.dumps(objects))

    def run_make(self, command, **settings):
        return subprocess.run(["make", "-s", "--no-print-directory", command], cwd=ROOT,
                              env={**self.env, **settings}, capture_output=True, text=True)

    def get(self, kind, name=None):
        args = [str(FAKE_OC), "get", kind]
        if name:
            args.append(name)
        result = subprocess.run([*args, "-n", NAMESPACE, "-o", "json"], env=self.env,
                                capture_output=True, text=True)
        return json.loads(result.stdout) if result.returncode == 0 else None

    def test_uninstall_removes_workloads_and_preserves_claim_and_namespace_resources(self):
        claim = self.get("pvc", CLAIM)
        result = self.run_make("uninstall")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.get("pvc", CLAIM), claim)
        self.assertIsNotNone(self.get("secret", "ibm-entitlement"))
        self.assertIsNotNone(self.get("configmap", "openshift-service-ca.crt"))
        self.assertIsNotNone(self.get("deployment", "unrelated"))
        self.assertEqual(self.get("pods"), {"items": []})
        for kind, name in (("inferenceservice", MODEL), ("route", MODEL),
                           ("servingruntime", "mistral-runtime"), ("job", "mistral-2411-prepare"),
                           ("configmap", "mistral-2411-preparation")):
            self.assertIsNone(self.get(kind, name))
        self.assertIn("preserved", result.stdout)

    def test_purge_removes_only_the_identified_claim_after_uninstall(self):
        stopped = self.run_make("uninstall")
        self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
        result = self.run_make("purge")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIsNone(self.get("pvc", CLAIM))
        self.assertIsNotNone(self.get("secret", "ibm-entitlement"))
        self.assertIsNotNone(self.get("deployment", "unrelated"))
        self.assertIn(f"namespace={NAMESPACE} PVC={CLAIM}", result.stdout)

    def test_both_commands_report_already_absent_on_repeat(self):
        for command in ("uninstall", "purge"):
            with self.subTest(command=command):
                first = self.run_make(command)
                second = self.run_make(command)
                self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
                self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
                self.assertIn("already absent", second.stdout)

    def test_purge_refuses_existing_workloads_and_preserves_claim(self):
        claim = self.get("pvc", CLAIM)
        result = self.run_make("purge")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run make uninstall first", result.stdout)
        self.assertEqual(self.get("pvc", CLAIM), claim)
        self.assertIsNotNone(self.get("inferenceservice", MODEL))

    def test_purge_refuses_any_pod_referencing_claim_even_after_uninstall(self):
        stopped = self.run_make("uninstall")
        self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
        for phase in ("Running", "Pending", "Succeeded", "Failed"):
            with self.subTest(phase=phase):
                objects = json.loads(self.state_path.read_text())
                objects["pod/other-consumer"] = resource(
                    "Pod", "other-consumer", status={"phase": phase},
                    spec={"volumes": [{"persistentVolumeClaim": {"claimName": CLAIM}}]})
                self.state_path.write_text(json.dumps(objects))
                result = self.run_make("purge")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("pods still reference PVC", result.stdout)
                self.assertIsNotNone(self.get("pvc", CLAIM))

    def test_uninstall_reports_orphaned_validation_pods_instead_of_success(self):
        objects = json.loads(self.state_path.read_text())
        pod = resource("Pod", "orphaned-predictor")
        pod["metadata"]["labels"] = {"serving.kserve.io/inferenceservice": MODEL}
        objects["pod/orphaned-predictor"] = pod
        self.state_path.write_text(json.dumps(objects))
        result = self.run_make("uninstall")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("validation pods still exist", result.stdout)
        self.assertIsNotNone(self.get("pvc", CLAIM))

    def test_access_and_deletion_errors_remain_failures_without_raw_server_output(self):
        for scenario in ("cleanup_denied", "cleanup_delete_failure"):
            with self.subTest(scenario=scenario):
                result = self.run_make("uninstall", FAKE_OC_SCENARIO=scenario)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("check access, events, and finalizers", result.stdout)
                self.assertNotIn("private-server-error", result.stdout + result.stderr)
                self.assertIsNotNone(self.get("pvc", CLAIM))
        stopped = self.run_make("uninstall")
        self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
        for scenario in ("cleanup_denied", "cleanup_delete_failure"):
            with self.subTest(scenario=scenario):
                result = self.run_make("purge", FAKE_OC_SCENARIO=scenario)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("private-server-error", result.stdout + result.stderr)
                self.assertIsNotNone(self.get("pvc", CLAIM))

    def test_invalid_cleanup_timeout_fails_before_deletion(self):
        result = self.run_make("uninstall", CLEANUP_TIMEOUT="0s")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CLEANUP_TIMEOUT must be a positive number of seconds", result.stdout)
        self.assertIsNotNone(self.get("inferenceservice", MODEL))


if __name__ == "__main__":
    unittest.main()
