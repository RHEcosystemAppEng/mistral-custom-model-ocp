"""Preview and execution share effective discovery and preparation resources."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_inventory import logs
from test_preparation import FAKE_OC, IMAGES


class PreparationRenderingTests(unittest.TestCase):
    def invoke(self, command, settings=(), env=None, **options):
        return subprocess.run(
            [sys.executable, "-m", "mistral_validation", command, *settings],
            env={**os.environ, **(env or {})}, capture_output=True, text=True, **options)

    def test_inventory_applies_the_offline_discovery_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = ["--namespace", "inventory-preview", "--pull-secret", "private-pull"]
            preview = self.invoke("render-discovery", settings + ["--oc", "/missing/oc"])
            self.assertEqual(preview.returncode, 0, preview.stderr)
            resources = json.loads(preview.stdout)
            result = self.invoke("inventory", settings + ["--oc", str(FAKE_OC), "--layout", str(root / "layout.json")], env={
                "FAKE_OC_SCENARIO": "inventory", "FAKE_OC_WEIGHT_IMAGES": json.dumps(IMAGES),
                "FAKE_OC_INVENTORY_LOGS": json.dumps(logs()),
                "FAKE_OC_APPLIED_RESOURCES": str(root / "applied.jsonl"),
                "FAKE_OC_MUTATIONS": str(root / "mutations.jsonl")})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            applied = [json.loads(line) for line in (root / "applied.jsonl").read_text().splitlines()]
            self.assertEqual(applied, [resources])
            mutations = [json.loads(line) for line in (root / "mutations.jsonl").read_text().splitlines()]
            self.assertEqual([item[0] for item in mutations], ["delete"] * 4 + ["apply"])
            for part, job in enumerate(resources["items"], 1):
                self.assertEqual(job["metadata"], {"name": f"mistral-2411-discover-{part}", "namespace": "inventory-preview"})
                pod = job["spec"]["template"]["spec"]
                self.assertEqual(pod["containers"][0]["image"], IMAGES[part - 1])
                self.assertEqual(pod["imagePullSecrets"], [{"name": "private-pull"}])

    def test_preparation_applies_preview_with_live_claim_identity(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                layout = {"model": "mistral-large-instruct-2411", "source_images": IMAGES,
                          "assembly": "overlay", "files": [
                              {"image": 4 if sparse else part, "source": f"/synthetic/a 'quote'/file{part}",
                               "target": f"nested: files/{{{{namespace}}}}/file{part}", "size": 1,
                               "sha256": "a" * 64} for part in range(1, 5)]}
                path = root / "layout.json"
                path.write_text(json.dumps(layout))
                settings = ["--layout", str(path), "--namespace", "preparation-preview",
                            "--pull-secret", "private-pull", "--pvc-name", "prepared-model"]
                if sparse:
                    settings += ["--storage-class", "fast", "--pvc-size", "200Gi"]
                preview = self.invoke("render-preparation", settings + ["--oc", "/missing/oc"])
                self.assertEqual(preview.returncode, 0, preview.stderr)
                resources = json.loads(preview.stdout)["items"]
                result = self.invoke("prepare", settings + ["--oc", str(FAKE_OC), "--report-dir", str(root / "reports")], env={
                    "FAKE_OC_SCENARIO": "preparation_success", "FAKE_OC_WEIGHT_IMAGES": json.dumps(IMAGES),
                    "FAKE_OC_SCENARIO_PVC": "no_pvc",
                    "FAKE_OC_PREPARATION_STATE": str(root / "state.json"),
                    "FAKE_OC_APPLIED_RESOURCES": str(root / "applied.jsonl"),
                    "FAKE_OC_MUTATIONS": str(root / "mutations.jsonl")})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                applied = [json.loads(line) for line in (root / "applied.jsonl").read_text().splitlines()]
                # Only the UID, unavailable offline, is bound after rendering.
                self.assertEqual(applied[2]["spec"]["template"].pop("metadata"), {
                    "annotations": {"mistral-validation/pvc-uid": "uid-prepared-model"}})
                self.assertEqual(applied, resources)
                mutations = [json.loads(line) for line in (root / "mutations.jsonl").read_text().splitlines()]
                self.assertEqual([item[0] for item in mutations], ["delete", "apply", "apply", "apply"])
                self.assertIn("--cascade=foreground", mutations[0])
                self.assertIn("--wait=true", mutations[0])
                pvc, configmap, job = resources
                self.assertEqual(pvc["spec"]["resources"]["requests"]["storage"], "200Gi" if sparse else "140Gi")
                self.assertEqual(pvc["spec"].get("storageClassName"), "fast" if sparse else None)
                if not sparse:
                    self.assertNotIn("storageClassName", pvc["spec"])
                self.assertEqual(json.loads(configmap["data"]["layout.json"]), layout)
                pod = job["spec"]["template"]["spec"]
                self.assertEqual(pod["imagePullSecrets"], [{"name": "private-pull"}])
                self.assertEqual([container["image"] for container in pod["initContainers"]], IMAGES)
                self.assertTrue(all(item["metadata"]["namespace"] == "preparation-preview" for item in resources))

    def test_invalid_inputs_fail_before_cluster_access(self):
        cases = [(command, settings, False)
                 for command in ("render-discovery", "inventory", "render-preparation", "prepare")
                 for settings in (["--namespace", "bad/name"], ["--pull-secret", "bad:secret"],
                                  ["--pvc-size", "1Gi"], ["--storage-class", "bad/class"])]
        cases += [(command, [], True) for command in ("render-preparation", "prepare")]
        cases += [(command, ["--prepare-timeout", "0s"], False) for command in ("inventory", "prepare")]
        for command, settings, unsafe in cases:
            with self.subTest(command=command, settings=settings, unsafe=unsafe), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                marker = root / "cluster-access"
                oc = root / "oc"
                oc.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 1\n")
                oc.chmod(0o755)
                path = root / "layout.json"
                path.write_text(json.dumps({
                    "model": "mistral-large-instruct-2411", "source_images": IMAGES,
                    "assembly": "overlay", "files": [{
                        "image": 1, "source": "/synthetic/file", "target": "../escape" if unsafe else "file",
                        "size": 1, "sha256": "a" * 64}]}))
                result = self.invoke(command, [*settings, "--layout", str(path), "--oc", str(oc)])
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(marker.exists(), result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
