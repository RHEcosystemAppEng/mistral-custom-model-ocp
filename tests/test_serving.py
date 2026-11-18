import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from test_preparation import IMAGES


ROOT = Path(__file__).resolve().parents[1]
FAKE_OC = ROOT / "tests" / "fake_oc.py"
RUNTIME_IMAGE = "cp.icr.io/cp/cpd/mistral-inference-server@sha256:03512d44a97adb6aff115e025df819ec05b249728471effe6a5615880af593a9"


class ServingWorkflowTests(unittest.TestCase):
    def run_make(self, target, **settings):
        env = {**os.environ, "NAMESPACE": "mistral-custom-model-ocp", "GPU_COUNT": "4",
               "PVC_NAME": "mistral-weights", "SERVING_TIMEOUT": "120s",
               "FAKE_OC_RUNTIME_IMAGE": RUNTIME_IMAGE, **settings}
        return subprocess.run(["make", "-s", "--no-print-directory", target], cwd=ROOT,
                              env=env, capture_output=True, text=True)

    def test_rendered_runtime_uses_prepared_weights_and_health_gates(self):
        result = self.run_make("render-deployment")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        runtime, service = json.loads(result.stdout)["items"]
        container = runtime["spec"]["containers"][0]
        self.assertIn("@sha256:03512d44a97adb6aff115e025df819ec05b249728471effe6a5615880af593a9", container["image"])
        self.assertEqual(container["command"], [
            "/bin/sh", "-c",
            'mkdir -p "$PROMETHEUS_MULTIPROC_DIR" && '
            'exec python3 -m vllm.entrypoints.openai.api_server "$@"',
            "mistral-runtime",
        ])
        self.assertEqual({item["name"]: item["value"] for item in container["env"]}, {
            "HOME": "/var/tmp/mistral-runtime-home",
            "TMPDIR": "/var/tmp",
            "PROMETHEUS_MULTIPROC_DIR": "/var/tmp/prometheus_metrics",
            "XDG_CACHE_HOME": "/var/tmp/mistral-runtime-cache",
            "VLLM_CACHE_ROOT": "/var/tmp/mistral-runtime-vllm",
            "TORCH_EXTENSIONS_DIR": "/var/tmp/mistral-runtime-torch",
            "TRITON_CACHE_DIR": "/var/tmp/.triton_cache",
            "TORCHINDUCTOR_CACHE_DIR": "/var/tmp/.torchinductor_cache",
            "VLLM_USE_FALLBACK_LOADER": "1",
        })
        self.assertIn("--tensor-parallel-size=4", container["args"])
        for argument in ("--model=/mnt/models/mistral-large-instruct-2411",
                         "--served-model-name=mistralai/mistral-large-instruct-2411",
                         "--port=3000", "--max-model-len=32768", "--max-num-seqs=20",
                         "--tool-call-parser=mistral", "--enable-auto-tool-choice",
                         "--disable-log-requests"):
            self.assertIn(argument, container["args"])
        self.assertEqual(container["ports"][0]["containerPort"], 3000)
        self.assertFalse(container["securityContext"]["allowPrivilegeEscalation"])
        self.assertEqual(container["startupProbe"]["httpGet"], {"path": "/health", "port": 3000})
        self.assertEqual(container["readinessProbe"]["httpGet"], {"path": "/health", "port": 3000})
        self.assertEqual(container["startupProbe"]["failureThreshold"] * container["startupProbe"]["periodSeconds"], 120)
        self.assertEqual(service["metadata"]["annotations"]["storage.kserve.io/readonly"], "true")
        self.assertEqual(service["metadata"]["annotations"]["security.opendatahub.io/enable-auth"], "true")
        self.assertEqual(service["metadata"]["labels"]["networking.kserve.io/visibility"], "exposed")
        self.assertEqual(service["spec"]["predictor"]["model"]["storageUri"], "pvc://mistral-weights/")
        self.assertEqual(service["spec"]["predictor"]["model"]["resources"]["limits"]["nvidia.com/gpu"], 4)
        self.assertEqual(service["spec"]["predictor"]["model"]["resources"]["requests"]["nvidia.com/gpu"], 4)
        self.assertEqual(service["spec"]["predictor"]["tolerations"],
                         [{"key": "p4-gpu", "operator": "Exists", "effect": "NoSchedule"}])

    def test_runtime_creates_metrics_directory_and_preserves_server_arguments(self):
        rendered = self.run_make("render-deployment")
        self.assertEqual(rendered.returncode, 0, rendered.stdout + rendered.stderr)
        container = json.loads(rendered.stdout)["items"][0]["spec"]["containers"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "bin"
            binary.mkdir()
            (binary / "python3").symlink_to(sys.executable)
            package = root
            for name in ("vllm", "entrypoints", "openai"):
                package = package / name
                package.mkdir()
                (package / "__init__.py").touch()
            (package / "api_server.py").write_text(
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "print(json.dumps({'metrics_exists': "
                "Path(os.environ['PROMETHEUS_MULTIPROC_DIR']).is_dir(), "
                "'args': sys.argv[1:]}))\n")
            arguments = container["args"] + ["--probe=literal $value with spaces"]
            result = subprocess.run(container["command"] + arguments, capture_output=True,
                                    text=True, cwd=root, env={
                                        **os.environ, "PATH": f"{binary}{os.pathsep}{os.environ['PATH']}",
                                        "PYTHONPATH": str(root),
                                        "PROMETHEUS_MULTIPROC_DIR": str(root / "metrics with spaces"),
                                    })
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            evidence = json.loads(result.stdout)
            self.assertTrue(evidence["metrics_exists"])
            self.assertEqual(evidence["args"], arguments)

    def test_install_requires_verified_preparation_before_applying(self):
        result = self.run_make("install", OC=str(FAKE_OC), FAKE_OC_SCENARIO="ready")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preparation", result.stdout.lower())
        self.assertIn("make prepare", result.stdout)

    def test_only_optional_attention_scales_may_be_missing(self):
        result = self.run_make("render-deployment")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = json.loads(result.stdout)["items"][0]["spec"]["containers"][0]["args"]
        self.assertIn("--kv-cache-dtype=auto", args)
        pattern = next(arg.split("=", 1)[1] for arg in args
                       if arg.startswith("--allowed-missing-params="))
        for scale in ("q", "k", "v", "prob"):
            self.assertIsNotNone(re.fullmatch(pattern, f"model.layers.87.self_attn.attn.{scale}_scale"))
        for name in ("model.layers.0.self_attn.q_proj.weight",
                     "model.layers.0.self_attn.q_proj.weight_scale",
                     "model.layers.0.self_attn.q_proj.input_scale",
                     "model.layers.0.mlp.gate_proj.weight", "lm_head.weight",
                     "modelXlayersX0Xself_attnXattnXk_scale"):
            self.assertIsNone(re.fullmatch(pattern, name), name)

    def test_runtime_check_reports_metadata_success_and_failure_without_raw_output(self):
        for scenario, expected in (("serving_success", "passed"),
                                   ("serving_metadata_failure", "failed")):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                result = self.run_make("runtime-check", OC=str(FAKE_OC), FAKE_OC_SCENARIO=scenario,
                                       FAKE_OC_MARKER=str(Path(directory) / "state"),
                                       FAKE_OC_WEIGHT_IMAGES=json.dumps(IMAGES), REPORT_DIR=directory)
                reports = list(Path(directory).glob("runtime-check-*.json"))
                self.assertEqual(len(reports), 1, result.stdout + result.stderr)
                evidence = json.loads(reports[0].read_text())
                self.assertEqual(evidence["result"], expected)
                self.assertEqual(result.returncode == 0, expected == "passed")
                self.assertTrue(evidence["metadata_only"])
                self.assertEqual(evidence["configured_tensor_parallel_size"], 4)
                self.assertEqual(evidence["checked_tensor_parallel_size"], 1)
                self.assertNotIn("private-", result.stdout + result.stderr + reports[0].read_text())

    def test_install_observes_loading_then_validates_inference_and_saves_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_make("install", OC=str(FAKE_OC), FAKE_OC_SCENARIO="serving_success",
                                   FAKE_OC_MARKER=str(Path(directory) / "state"), REPORT_DIR=directory,
                                   FAKE_OC_WEIGHT_IMAGES=json.dumps(IMAGES), SERVING_TIMEOUT="15s")
            reports = list(Path(directory).glob("serving-*.json"))
            self.assertEqual(len(reports), 1, result.stdout + result.stderr)
            report_text = reports[0].read_text()
            evidence = json.loads(report_text)
            self.assertNotIn("Reply with one word", report_text)
            self.assertNotIn("Ready.", report_text)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(evidence["observed_unready"])
        self.assertTrue(evidence["observed_health_unready"])
        self.assertTrue(evidence["health_ready"])
        self.assertTrue(evidence["inference_valid"])
        self.assertGreaterEqual(evidence["model_load_seconds"], 0)
        self.assertGreaterEqual(evidence["install_to_ready_seconds"], evidence["model_load_seconds"])
        self.assertEqual(evidence["served_model"], "mistralai/mistral-large-instruct-2411")

    def test_health_alone_does_not_count_as_successful_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_make("install", OC=str(FAKE_OC), FAKE_OC_SCENARIO="serving_bad_inference",
                                   FAKE_OC_MARKER=str(Path(directory) / "state"), REPORT_DIR=directory,
                                   FAKE_OC_WEIGHT_IMAGES=json.dumps(IMAGES), SERVING_TIMEOUT="15s")
            reports = list(Path(directory).glob("serving-*.json"))
            self.assertEqual(len(reports), 1, result.stdout + result.stderr)
            evidence = json.loads(reports[0].read_text())
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(evidence["health_ready"])
        self.assertFalse(evidence["inference_valid"])

    def test_status_reports_pod_and_service_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {"OC": str(FAKE_OC), "FAKE_OC_SCENARIO": "serving_pod_loading",
                   "FAKE_OC_MARKER": str(Path(directory) / "state"),
                   "FAKE_OC_WEIGHT_IMAGES": json.dumps(IMAGES)}
            first = self.run_make("status", **env)
            second = self.run_make("status", **env)
        self.assertNotEqual(first.returncode, 0)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("predictor pod", first.stdout)

    def test_scc_rejection_is_recorded_as_ibm_integration_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_make("install", OC=str(FAKE_OC), FAKE_OC_SCENARIO="serving_scc_denied",
                                   FAKE_OC_MARKER=str(Path(directory) / "state"), REPORT_DIR=directory,
                                   FAKE_OC_WEIGHT_IMAGES=json.dumps(IMAGES), SERVING_TIMEOUT="1s")
            reports = list(Path(directory).glob("serving-*.json"))
            self.assertEqual(len(reports), 1, result.stdout + result.stderr)
            evidence = json.loads(reports[0].read_text())
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "ibm_integration_gap")
        self.assertIn("IBM integration gap", result.stdout)

    def test_install_rejects_a_writable_model_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_make("install", OC=str(FAKE_OC), FAKE_OC_SCENARIO="serving_writable_mount",
                                   FAKE_OC_MARKER=str(Path(directory) / "state"), REPORT_DIR=directory,
                                   FAKE_OC_WEIGHT_IMAGES=json.dumps(IMAGES), SERVING_TIMEOUT="8s")
            reports = list(Path(directory).glob("serving-*.json"))
            self.assertEqual(len(reports), 1, result.stdout + result.stderr)
            evidence = json.loads(reports[0].read_text())
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "mount_invalid")

    def test_elevated_scc_does_not_count_as_verified_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_make("install", OC=str(FAKE_OC), FAKE_OC_SCENARIO="serving_elevated_scc",
                                   FAKE_OC_MARKER=str(Path(directory) / "state"), REPORT_DIR=directory,
                                   FAKE_OC_WEIGHT_IMAGES=json.dumps(IMAGES), SERVING_TIMEOUT="8s")
            reports = list(Path(directory).glob("serving-*.json"))
            self.assertEqual(len(reports), 1, result.stdout + result.stderr)
            evidence = json.loads(reports[0].read_text())
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(evidence["result"], "ibm_integration_gap")

    def test_install_does_not_claim_transition_if_health_was_already_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_make("install", OC=str(FAKE_OC), FAKE_OC_SCENARIO="serving_no_transition",
                                   FAKE_OC_MARKER=str(Path(directory) / "state"), REPORT_DIR=directory,
                                   FAKE_OC_WEIGHT_IMAGES=json.dumps(IMAGES), SERVING_TIMEOUT="15s")
            reports = list(Path(directory).glob("serving-*.json"))
            self.assertEqual(len(reports), 1, result.stdout + result.stderr)
            evidence = json.loads(reports[0].read_text())
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(evidence["observed_health_unready"])
        self.assertEqual(evidence["result"], "transition_not_observed")


if __name__ == "__main__":
    unittest.main()
