"""An installed distribution works without the source checkout on its import path."""

import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import venv


ROOT = Path(__file__).resolve().parents[1]


class InstalledCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.outside = Path(cls.temporary.name)
        environment = cls.outside / "environment"
        venv.create(environment, with_pip=True)
        cls.python = environment / "bin" / "python"
        cls.cli = environment / "bin" / "mistral-validate"
        cls.env = dict(os.environ)
        cls.env.pop("PYTHONPATH", None)
        cls.env.pop("PYTHONHOME", None)
        result = subprocess.run(
            [str(cls.python), "-m", "pip", "install", str(ROOT)],
            cwd=cls.outside, env=cls.env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def invoke(self, *args):
        return subprocess.run([str(self.cli), *args], cwd=self.outside,
                              env=self.env, capture_output=True, text=True)

    def test_help_and_offline_render_outside_checkout(self):
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("render-deployment", result.stdout)
        result = self.invoke("render-deployment", "--namespace", "installed-cli")
        self.assertEqual(result.returncode, 0, result.stderr)
        resources = json.loads(result.stdout)["items"]
        self.assertTrue(resources)
        self.assertTrue(all(item["metadata"]["namespace"] == "installed-cli"
                            for item in resources))

    def test_installed_yaml_is_authoritative_for_runtime_preview(self):
        result = subprocess.run(
            [str(self.python), "-c",
             "from importlib.resources import files; "
             "print(files('mistral_validation').joinpath('templates/serving-runtime.yaml'))"],
            cwd=self.outside, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        template = Path(result.stdout.strip())
        original = template.read_text()
        try:
            template.write_text(original.replace("--max-num-seqs=20", "--max-num-seqs=12"))
            result = self.invoke("render-deployment")
            self.assertEqual(result.returncode, 0, result.stderr)
            container = json.loads(result.stdout)["items"][0]["spec"]["containers"][0]
            self.assertIn("--max-num-seqs=12", container["args"])
            self.assertNotIn("--max-num-seqs=20", container["args"])
        finally:
            template.write_text(original)

    def test_preparation_includes_a_worker_that_verifies_and_publishes(self):
        from test_preparation import IMAGES

        root = self.outside / "preparation"
        root.mkdir()
        stage = root / ".mistral-2411-staging"
        stage.mkdir()
        content = b"installed worker test data"
        (stage / "model.safetensors").write_bytes(content)
        (root / ".mistral-2411-peak-kib").write_text("1")
        layout = root / "layout.json"
        layout.write_text(json.dumps({
            "model": "mistral-large-instruct-2411", "source_images": IMAGES,
            "assembly": "overlay", "files": [{
                "image": 1, "source": "/weights/model.safetensors", "target": "model.safetensors",
                "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}]}))
        result = self.invoke("render-preparation", "--layout", str(layout))
        self.assertEqual(result.returncode, 0, result.stderr)
        configmap = next(item for item in json.loads(result.stdout)["items"]
                         if item["kind"] == "ConfigMap")
        worker = root / "worker.py"
        worker.write_text(configmap["data"]["prepare_worker.py"])
        result = subprocess.run([str(self.python), str(worker)], cwd=self.outside,
                                env={**self.env, "PREPARATION_ROOT": str(root),
                                     "PREPARATION_LAYOUT": str(layout)},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((root / "mistral-large-instruct-2411" / "model.safetensors").read_bytes(),
                         content)

    def test_runtime_diagnostic_resource_is_usable_without_the_package(self):
        # The cluster receives this module as a standalone stdin program.
        result = subprocess.run(
            [str(self.python), "-c",
             "from importlib.resources import files; "
             "print(files('mistral_validation').joinpath('runtime_check.py').read_text())"],
            cwd=self.outside, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([str(self.python), "-I", "-", "[]", "{}"], input=result.stdout,
                                cwd=self.outside, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        summary = next(line.removeprefix("RUNTIME_CHECK ") for line in result.stdout.splitlines()
                       if line.startswith("RUNTIME_CHECK "))
        self.assertEqual(json.loads(summary)["exception_type"], "ModuleNotFoundError")
        self.assertEqual(json.loads(summary)["stage"], "imports")

    def test_runbook_children_use_the_installed_environment(self):
        root = self.outside / "runbook"
        root.mkdir()
        marker = root / "child-preflight"
        oc = root / "oc"
        oc.write_text(
            f"#!{self.python}\n"
            "from pathlib import Path\nimport sys\n"
            "if sys.argv[1:3] in (['get', 'pvc'], ['get', 'inferenceservice']):\n"
            "    print('Error from server (NotFound)', file=sys.stderr)\n"
            "elif sys.argv[1:] == ['whoami']:\n"
            f"    Path({str(marker)!r}).touch()\n"
            "raise SystemExit(1)\n")
        oc.chmod(0o755)
        # A bare python3 subprocess would fail before reaching the fake oc.
        wrong_python = root / "python3"
        wrong_python.write_text("#!/bin/sh\nexit 99\n")
        wrong_python.chmod(0o755)
        result = subprocess.run(
            [str(self.cli), "verify-run", "--oc", str(oc), "--report-dir", str(root)],
            cwd=self.outside, env={**self.env, "PATH": str(root), "RHOAI_TOKEN": "test-token"},
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertTrue(marker.exists(), result.stdout + result.stderr)
        report = json.loads(next(root.glob("run-*/verification.json")).read_text())
        self.assertEqual(report["checks"]["preflight"]["result"], "failed")


if __name__ == "__main__":
    unittest.main()
