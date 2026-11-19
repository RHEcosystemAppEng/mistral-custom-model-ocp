"""Check the pinned runtime's tensor mapping using headers and meta tensors."""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import types
from typing import Any

MARKER = "RUNTIME_CHECK "


def read_header(stream):
    length_bytes = stream.read(8)
    if len(length_bytes) != 8:
        raise ValueError("truncated safetensors header length")
    length = struct.unpack("<Q", length_bytes)[0]
    if not 0 < length <= 64 * 1024**2:
        raise ValueError("invalid safetensors header length")
    encoded = stream.read(length)
    if len(encoded) != length:
        raise ValueError("truncated safetensors header")
    header = json.loads(encoded)
    if not isinstance(header, dict):
        raise ValueError("invalid safetensors header")
    return header


def metadata_main(runtime_args, runtime_env):
    """Executed inside the pinned image; never reads tensor payload bytes."""
    import os
    os.environ.update(runtime_env)
    start = time.monotonic()
    evidence: dict[str, Any] = {"metadata_only": True, "checked_tensor_parallel_size": 1,
                "shards": 0, "tensors": 0, "stage": "imports"}
    try:
        import torch
        from vllm.engine.arg_utils import EngineArgs
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        from vllm.distributed import init_distributed_environment, initialize_model_parallel
        from vllm.model_executor.model_loader.utils import initialize_model
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        from vllm.utils.torch_utils import set_default_torch_dtype

        torch.set_num_threads(1)
        evidence["stage"] = "configuration"
        parsed = make_arg_parser(FlexibleArgumentParser()).parse_args(runtime_args)
        args = EngineArgs.from_cli_args(parsed)
        args.skip_tokenizer_init = True
        args.disable_custom_all_reduce = True
        args.enforce_eager = True
        configured = args.create_engine_config()
        evidence["configured_tensor_parallel_size"] = configured.parallel_config.tensor_parallel_size
        # First validate the intended parallel configuration, then perform the
        # shape/mapping check on one meta rank without allocating model weights.
        args.tensor_parallel_size = 1
        cfg = args.create_engine_config()
        evidence["stage"] = "meta_model"
        with tempfile.TemporaryDirectory(dir="/var/tmp") as directory:
            init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                         distributed_init_method=f"file://{directory}/store", backend="gloo")
            initialize_model_parallel(1, 1, backend="gloo")
            with set_default_torch_dtype(cfg.model_config.dtype), torch.device("meta"):
                model = initialize_model(vllm_config=cfg)

            def metadata_iterator(loader, source):
                loader.counter_before_loading_weights = time.perf_counter()
                _, files, _ = loader._prepare_weights(source.model_or_path, source.revision,
                                                      source.fall_back_to_pt, source.allow_patterns_overrides)
                dtypes = {"F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
                          "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
                          "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
                          "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
                for filename in files:
                    with open(filename, "rb") as stream:
                        header = read_header(stream)
                    evidence["shards"] += 1
                    for name, metadata in header.items():
                        if name != "__metadata__":
                            evidence["tensors"] += 1
                            yield name, torch.empty(metadata["shape"], dtype=dtypes[metadata["dtype"]],
                                                    device="meta")

            evidence["stage"] = "tensor_mapping"
            loader = DefaultModelLoader(cfg.load_config)
            loader._get_weights_iterator = types.MethodType(metadata_iterator, loader)
            loader.load_weights(model, cfg.model_config)
            evidence.update(result="passed", stage="complete")
    except (Exception, SystemExit) as error:
        # Runtime exceptions can contain private tensor names. Report types and
        # stages only; never echo their messages or the runtime's other output.
        evidence.update(result="failed", exception_type=type(error).__name__)
        if str(error).startswith("Following weights were not initialized"):
            evidence["failure_kind"] = "missing_parameters"
    finally:
        if "torch" in locals() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    evidence["elapsed_seconds"] = round(time.monotonic() - start, 3)
    print(MARKER + json.dumps(evidence), flush=True)
    return 0 if evidence["result"] == "passed" else 1


def check(oc, cfg, container, report_dir, oc_error):
    from .serving import MODEL, mounted_readonly, save_report

    namespace = cfg["namespace"]
    try:
        pods = oc.get("pods", "-n", namespace,
                      "-l", f"serving.kserve.io/inferenceservice={MODEL}").get("items", [])
        candidates = [pod for pod in pods if mounted_readonly(pod, cfg["pvc"]["name"], container["image"])
                      and any(status.get("name") == "kserve-container" and "running" in status.get("state", {})
                              for status in pod.get("status", {}).get("containerStatuses", []))]
        if not candidates:
            print("FAIL runtime-check: requires a running pinned runtime container with the model PVC mounted read-only")
            return 1
        pod = max(candidates, key=lambda item: item.get("metadata", {}).get("creationTimestamp", ""))
        result = subprocess.run(
            [oc.binary, "exec", "-i", "-n", namespace, pod["metadata"]["name"], "-c", "kserve-container",
             "--", "python3", "-", json.dumps(container["args"]),
             json.dumps({item["name"]: item["value"] for item in container["env"]})],
            input=Path(__file__).read_text(), capture_output=True, text=True, timeout=180, check=False)
        summaries = [line[len(MARKER):] for line in result.stdout.splitlines() if line.startswith(MARKER)]
        evidence = json.loads(summaries[-1]) if summaries else {"result": "failed", "stage": "exec"}
        passed = result.returncode == 0 and evidence.get("result") == "passed"
        evidence.update(result="passed" if passed else "failed", metadata_only=True,
                        runtime_image=container["image"], gpu_count=cfg["gpu_count"])
        path = save_report(report_dir, namespace, evidence, prefix="runtime-check")
    except (oc_error, OSError, subprocess.TimeoutExpired, ValueError):
        print("FAIL runtime-check: check pod exec access, runtime configuration, and REPORT_DIR")
        return 1
    print(f"{'PASS' if passed else 'FAIL'} runtime-check: tensor metadata only; "
          f"stage={evidence.get('stage')}; elapsed={evidence.get('elapsed_seconds')}s; report {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(metadata_main(json.loads(sys.argv[1]), json.loads(sys.argv[2])))
