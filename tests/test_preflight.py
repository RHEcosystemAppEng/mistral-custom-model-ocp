import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FAKE_OC = ROOT / "tests" / "fake_oc.py"


class OperatorWorkflowTests(unittest.TestCase):
    def run_make(self, target="preflight", scenario="ready", gpu_count="8", extra=None):
        env = os.environ.copy()
        env.update({"FAKE_OC_SCENARIO": scenario, "NAMESPACE": "mistral-custom-model-ocp", "GPU_COUNT": gpu_count,
                    "PULL_SECRET": "ibm-entitlement", "PVC_NAME": "mistral-weights", "PVC_SIZE": "140Gi",
                    "STORAGE_CLASS": "", "OC": str(FAKE_OC)})
        if extra:
            env.update(extra)
        with tempfile.TemporaryDirectory() as directory:
            env["FAKE_OC_MARKER"] = str(Path(directory) / "created")
            return subprocess.run(["make", "--no-print-directory", target], cwd=ROOT, env=env, text=True, capture_output=True)

    def test_ready_cluster_passes_without_exposing_secret_data(self):
        result = self.run_make()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Preflight passed", result.stdout)
        self.assertNotIn("never-print-this", result.stdout + result.stderr)

    def test_gpu_count_requires_room_on_one_reachable_node(self):
        fail = self.run_make(scenario="busy_gpu")
        self.assertNotEqual(fail.returncode, 0)
        self.assertIn("need 8 free nvidia.com/gpu on one Ready node", fail.stdout)
        self.assertIn("gpu-1: 4/8 free", fail.stdout)
        success = self.run_make(scenario="busy_gpu", gpu_count="4")
        self.assertEqual(success.returncode, 0, success.stdout + success.stderr)

    def test_other_gpu_taint_is_actionable(self):
        result = self.run_make(scenario="tainted_gpu")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("untolerated taint special", result.stdout)

    def test_namespace_gpu_quota_is_checked(self):
        result = self.run_make(scenario="small_quota")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("team-quota has fewer than 8 free requests.nvidia.com/gpu", result.stdout)

    def test_missing_secret_and_kserve_capability_are_reported(self):
        for scenario, message in (("no_secret", "create existing registry pull secret"),
                                  ("no_kserve", "inferenceservices.serving.kserve.io: CRD is missing")):
            with self.subTest(scenario=scenario):
                result = self.run_make(scenario=scenario)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout)

    def test_absent_pvc_can_be_created_later(self):
        result = self.run_make(scenario="no_pvc")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PVC: absent;", result.stdout)

    def test_static_storage_without_available_volume_fails(self):
        result = self.run_make(scenario="static_storage", extra={"FAKE_OC_SCENARIO_PVC": "no_pvc"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no matching Available persistent volume", result.stdout)

    def test_unavailable_kserve_controller_fails(self):
        result = self.run_make(scenario="controller_down")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("kserve-controller-manager has no available replicas", result.stdout)

    def test_rendered_configuration_uses_one_gpu_count(self):
        result = self.run_make(target="render-config", gpu_count="4")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        document = json.loads(result.stdout[result.stdout.index("{"):])
        self.assertEqual(document["gpu_count"], 4)
        self.assertEqual(document["serving"]["gpu_resources"]["requests"]["nvidia.com/gpu"], 4)
        self.assertEqual(document["serving"]["gpu_resources"]["limits"]["nvidia.com/gpu"], 4)
        self.assertEqual(document["serving"]["tensor_parallel_size"], 4)
        self.assertEqual(len(document["images"]["weights"]), 4)
        self.assertTrue(all("@sha256:" in image for image in document["images"]["weights"]))

    def test_invalid_configuration_fails_before_cluster_access(self):
        result = self.run_make(gpu_count="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU_COUNT must be a positive integer", result.stderr)
        result = self.run_make(extra={"PVC_SIZE": "139Gi"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("140Gi baseline", result.stderr)


if __name__ == "__main__":
    unittest.main()
