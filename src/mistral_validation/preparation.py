"""Render the digest-pinned, inventory-gated model preparation resources."""

from importlib.resources import files
import json
from pathlib import Path, PurePosixPath
import re
import shlex

from .resource_templates import read_template, substitute


MODEL = "mistral-large-instruct-2411"
JOB_NAME = "mistral-2411-prepare"
CONFIG_NAME = "mistral-2411-preparation"
PVC_UID_ANNOTATION = "mistral-validation/pvc-uid"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def read_layout(path, weight_images):
    if not path:
        raise ValueError("LAYOUT must point to the 2411 artifact inventory JSON; run make inventory")
    try:
        layout = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read LAYOUT JSON at {path}; run make inventory") from error
    if not isinstance(layout, dict):
        raise ValueError("LAYOUT must be a JSON object")
    if layout.get("model") != MODEL:
        raise ValueError(f"LAYOUT model must be {MODEL}")
    if layout.get("source_images") != weight_images:
        raise ValueError("LAYOUT source_images must exactly match the four pinned 2411 images")
    if layout.get("assembly") != "overlay":
        raise ValueError("LAYOUT assembly must be overlay; shard concatenation is not implemented")
    files = layout.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("LAYOUT files must list the inventory")
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("LAYOUT files entries must be objects")
        part = entry.get("image")
        if type(part) is not int or part not in (1, 2, 3, 4):
            raise ValueError("LAYOUT image must be an integer from 1 to 4")
        source = entry.get("source")
        target = entry.get("target")
        if not isinstance(source, str) or not source.startswith("/") or any(piece in ("", ".", "..") for piece in source.split("/")[1:]):
            raise ValueError("LAYOUT source must be a normalized absolute file path")
        if not isinstance(target, str) or target.startswith("/") or any(piece in ("", ".", "..") for piece in target.split("/")) or target.startswith(".prep") or target == ".complete.json":
            raise ValueError("LAYOUT target must be a safe relative file path")
        if any(target == other or target.startswith(other + "/") or other.startswith(target + "/") for other in seen):
            raise ValueError(f"LAYOUT target conflicts with another file: {target}")
        seen.add(target)
        if type(entry.get("size")) is not int or entry["size"] < 0:
            raise ValueError("LAYOUT size must be a nonnegative byte count")
        if not isinstance(entry.get("sha256"), str) or not SHA256.fullmatch(entry["sha256"]):
            raise ValueError("LAYOUT sha256 must be a lowercase 64-character hex digest")
    return layout


def copy_script(entries, first):
    lines = ["set -eu", "stage=/pvc/.mistral-2411-staging", "final=/pvc/mistral-large-instruct-2411"]
    if first:
        lines += ["if [ -e \"$final\" ]; then echo 'MODEL_ALREADY_COMPLETE: stop serving and inspect the completed directory before preparing again'; exit 1; fi",
                  "rm -rf \"$stage\" /pvc/.mistral-2411-peak-kib",
                  "mkdir -p \"$stage\""]
    lines += ["record_peak() {",
              "  used=$(du -sk /pvc | awk '{print $1}')",
              "  previous=$(cat /pvc/.mistral-2411-peak-kib 2>/dev/null || echo 0)",
              "  if [ \"$used\" -gt \"$previous\" ]; then echo \"$used\" > /pvc/.mistral-2411-peak-kib; fi",
              "}", "record_peak"]
    for entry in entries:
        source = shlex.quote(entry["source"])
        target = shlex.quote("/pvc/.mistral-2411-staging/" + entry["target"])
        parent = shlex.quote(str(PurePosixPath("/pvc/.mistral-2411-staging") / entry["target"]).rsplit("/", 1)[0])
        lines += [f"test -f {source} || {{ echo 'SOURCE_FILE_MISSING: inspect LAYOUT and the pinned image'; exit 1; }}",
                  f"mkdir -p {parent}",
                  f"if ! cp {source} {target}; then",
                  "  record_peak",
                  "  available=$(df -Pk /pvc | awk 'NR==2 {print $4}')",
                  "  if [ \"$available\" -lt 1024 ]; then echo 'PVC_CAPACITY_INSUFFICIENT: expand the PVC or increase PVC_SIZE, then retry';",
                  "  else echo 'COPY_FAILED: inspect source image and PVC, then retry'; fi",
                  "  exit 1",
                  "fi",
                  "record_peak"]
    return "\n".join(lines)


def render_discovery(cfg, weight_images):
    template = read_template("discovery-job")
    bindings = {"namespace": cfg["namespace"], "pull_secret": cfg["pull_secret"]}
    jobs = [substitute(template, {**bindings, "part": part, "image": image})
            for part, image in enumerate(weight_images, 1)]
    return {"apiVersion": "v1", "kind": "List", "items": jobs}


def render(cfg, layout, weight_images):
    bindings = {"namespace": cfg["namespace"], "pull_secret": cfg["pull_secret"],
                "pvc_name": cfg["pvc"]["name"], "pvc_size": cfg["pvc"]["size"],
                "storage_class": cfg["pvc"]["storage_class"],
                "worker": files("mistral_validation").joinpath("prepare_worker.py").read_text(),
                "layout": json.dumps(layout, separators=(",", ":"))}
    pvc = substitute(read_template("preparation-pvc"), bindings)
    if not cfg["pvc"]["storage_class"]:
        # Omission requests the cluster default; an empty string disables it.
        pvc["spec"].pop("storageClassName", None)
    init_template = read_template("preparation-init-container")
    init = []
    for part, image in enumerate(weight_images, 1):
        entries = [entry for entry in layout["files"] if entry["image"] == part]
        init.append(substitute(init_template, {"part": part, "image": image,
                                              "copy_script": copy_script(entries, part == 1)}))
    bindings["init_containers"] = init
    configmap = substitute(read_template("preparation-configmap"), bindings)
    job = substitute(read_template("preparation-job"), bindings)
    return {"apiVersion": "v1", "kind": "List", "items": [pvc, configmap, job]}
