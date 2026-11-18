import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FAKE_OC = ROOT / "tests" / "fake_oc.py"
IMAGES = [
    "cp.icr.io/cp/cpd/mistral-large-instruct-2411-part1@sha256:cf6e5d18c42a9bb54784b74829ca04bd3b7c9f6db262b7b70c2fad6fa260cdbb",
    "cp.icr.io/cp/cpd/mistral-large-instruct-2411-part2@sha256:1fc2d41306d18ff2111d6d07805c06b170a2827ccd10ee5e6e10dad6a757a483",
    "cp.icr.io/cp/cpd/mistral-large-instruct-2411-part3@sha256:188ffd7990437409be9f66cf35e84ac34b87115a8edf553dca629339db15be51",
    "cp.icr.io/cp/cpd/mistral-large-instruct-2411-part4@sha256:b1465deaeb1f8beb13fa71299ab6b5751678dfcdbae64ec989a779d9052ae3ce",
]


class PreparationInterfaceTests(unittest.TestCase):
    def run_command(self, target, layout=None, extra=None):
        env = os.environ.copy()
        if layout is not None:
            env["LAYOUT"] = str(layout)
        if extra:
            env.update(extra)
        return subprocess.run(["make", "--no-print-directory", target], cwd=ROOT,
                              env=env, capture_output=True, text=True)

    def test_preparation_requires_local_layout_before_cluster_changes(self):
        result = self.run_command("prepare", Path("/tmp/missing-2411-layout.json"), {"OC": str(FAKE_OC)})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("/tmp/missing-2411-layout.json", result.stderr)
        self.assertIn("make inventory", result.stderr)

    def test_rendered_job_uses_pinned_images_secret_and_pvc(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = Path(directory) / "layout.json"
            layout.write_text(json.dumps({"model": "mistral-large-instruct-2411", "source_images": IMAGES,
                                          "assembly": "overlay",
                                          "files": [{"image": part, "source": f"/weights/file{part}",
                                                     "target": f"file{part}", "size": 1,
                                                     "sha256": "a" * 64} for part in range(1, 5)]}))
            result = self.run_command("render-preparation", layout)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        objects = json.loads(result.stdout[result.stdout.index("{"):])["items"]
        job = next(item for item in objects if item["kind"] == "Job")
        pod = job["spec"]["template"]["spec"]
        self.assertEqual(pod["imagePullSecrets"], [{"name": "ibm-entitlement"}])
        self.assertEqual(len(pod["initContainers"]), 4)
        self.assertTrue(all("@sha256:" in part["image"] for part in pod["initContainers"]))
        self.assertEqual(pod["volumes"][0]["persistentVolumeClaim"]["claimName"], "mistral-weights")

    def test_unsafe_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = Path(directory) / "layout.json"
            layout.write_text(json.dumps({"model": "mistral-large-instruct-2411", "source_images": IMAGES,
                                          "assembly": "overlay",
                                          "files": [{"image": 1, "source": "/weights/file",
                                                     "target": "../escape", "size": 1,
                                                     "sha256": "a" * 64}]}))
            result = self.run_command("render-preparation", layout)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("target", result.stderr)

    def test_known_inventory_larger_than_pvc_fails_with_capacity_action(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = Path(directory) / "layout.json"
            layout.write_text(json.dumps({"model": "mistral-large-instruct-2411", "source_images": IMAGES,
                                          "assembly": "overlay",
                                          "files": [{"image": part, "source": f"/weights/file{part}",
                                                     "target": f"file{part}", "size": 40 * 1024**3,
                                                     "sha256": "a" * 64} for part in range(1, 5)]}))
            result = self.run_command("prepare", layout)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("increase PVC_SIZE", result.stdout)

    def test_failed_verification_keeps_final_hidden_and_retry_can_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / ".mistral-2411-staging"
            stage.mkdir()
            (root / ".mistral-2411-peak-kib").write_text("1")
            expected = b"verified model data"
            layout = root / "layout.json"
            layout.write_text(json.dumps({"model": "mistral-large-instruct-2411",
                                          "source_images": IMAGES,
                                          "files": [{"target": "model.safetensors", "size": len(expected),
                                                     "sha256": hashlib.sha256(expected).hexdigest()}]}))
            env = {**os.environ, "PREPARATION_ROOT": str(root), "PREPARATION_LAYOUT": str(layout)}
            (stage / "model.safetensors").write_bytes(b"interrupted")
            failed = subprocess.run(["python3", str(ROOT / "scripts" / "prepare_worker.py")],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((root / "mistral-large-instruct-2411").exists())
            (stage / "model.safetensors").write_bytes(expected)
            success = subprocess.run(["python3", str(ROOT / "scripts" / "prepare_worker.py")],
                                     env=env, capture_output=True, text=True)
            self.assertEqual(success.returncode, 0, success.stdout + success.stderr)
            final = root / "mistral-large-instruct-2411"
            self.assertEqual((final / "model.safetensors").read_bytes(), expected)
            self.assertEqual(json.loads((final / ".complete.json").read_text())["verified_files"], 1)

    def test_successful_prepare_records_observed_image_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = root / "layout.json"
            layout.write_text(json.dumps({"model": "mistral-large-instruct-2411", "source_images": IMAGES,
                                          "assembly": "overlay",
                                          "files": [{"image": part, "source": f"/weights/file{part}",
                                                     "target": f"file{part}", "size": 1,
                                                     "sha256": "a" * 64} for part in range(1, 5)]}))
            result = self.run_command("prepare", layout, {"OC": str(FAKE_OC),
                                                           "FAKE_OC_SCENARIO": "preparation_success",
                                                           "FAKE_OC_WEIGHT_IMAGES": json.dumps(IMAGES),
                                                           "REPORT_DIR": str(root / "reports")})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            reports = list((root / "reports").glob("*.json"))
            self.assertEqual(len(reports), 1)
            evidence = json.loads(reports[0].read_text())
            self.assertEqual(evidence["verified_files"], 4)
            self.assertEqual(len(evidence["source_images"]), 4)
            self.assertEqual(evidence["source_images"][0]["observed_image_id"], "observed-image-1")


if __name__ == "__main__":
    unittest.main()
