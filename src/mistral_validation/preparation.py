"""Render the digest-pinned, inventory-gated model preparation resources."""

import json
from pathlib import Path, PurePosixPath
import re
import shlex


MODEL = "mistral-large-instruct-2411"
JOB_NAME = "mistral-2411-prepare"
CONFIG_NAME = "mistral-2411-preparation"
PVC_UID_ANNOTATION = "mistral-validation/pvc-uid"
HELPER_IMAGE = "registry.access.redhat.com/ubi9/python-312:latest"
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
    root = f"/models/{MODEL}"
    script = ("set -eu\n"
              f"test -d {root} || {{ echo INVENTORY_ROOT_MISSING; exit 1; }}\n"
              f"find {root} -type f -exec sh -ec 'for file do "
              "bytes=$(wc -c < \"$file\"); digest=$(sha256sum \"$file\" | cut -d \" \" -f 1); "
              "printf \"FILE\\t%s\\t%s\\t%s\\n\" \"$file\" \"$bytes\" \"$digest\"; "
              "done' sh {} +\n"
              f"if [ -f {root}/model.safetensors.index.json ]; then\n"
              f"  printf 'INDEX\\t'; base64 < {root}/model.safetensors.index.json | tr -d '\\n'; printf '\\n'\n"
              "fi\n")
    jobs = []
    for part, image in enumerate(weight_images, 1):
        jobs.append({"apiVersion": "batch/v1", "kind": "Job",
                     "metadata": {"name": f"mistral-2411-discover-{part}", "namespace": cfg["namespace"]},
                     "spec": {"backoffLimit": 0, "template": {"spec": {
                         "restartPolicy": "Never", "imagePullSecrets": [{"name": cfg["pull_secret"]}],
                         "containers": [{"name": "inventory", "image": image, "imagePullPolicy": "Always",
                                         "command": ["/bin/sh", "-ec", script]}]}}}})
    return {"apiVersion": "v1", "kind": "List", "items": jobs}


def render(cfg, layout, weight_images):
    pvc_spec = {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": cfg["pvc"]["size"]}}}
    if cfg["pvc"]["storage_class"]:
        pvc_spec["storageClassName"] = cfg["pvc"]["storage_class"]
    pvc = {"apiVersion": "v1", "kind": "PersistentVolumeClaim",
           "metadata": {"name": cfg["pvc"]["name"], "namespace": cfg["namespace"]}, "spec": pvc_spec}
    worker = Path(__file__).with_name("prepare_worker.py").read_text()
    configmap = {"apiVersion": "v1", "kind": "ConfigMap",
                 "metadata": {"name": CONFIG_NAME, "namespace": cfg["namespace"]},
                 "data": {"prepare_worker.py": worker, "layout.json": json.dumps(layout, separators=(",", ":"))}}
    volume_mount = {"name": "model", "mountPath": "/pvc"}
    init = []
    for part, image in enumerate(weight_images, 1):
        entries = [entry for entry in layout["files"] if entry["image"] == part]
        init.append({"name": f"part-{part}", "image": image, "imagePullPolicy": "Always",
                     "command": ["/bin/sh", "-ec", copy_script(entries, part == 1)],
                     "volumeMounts": [volume_mount]})
    job = {"apiVersion": "batch/v1", "kind": "Job",
           "metadata": {"name": JOB_NAME, "namespace": cfg["namespace"]},
           "spec": {"backoffLimit": 0, "template": {"spec": {
               "restartPolicy": "Never", "imagePullSecrets": [{"name": cfg["pull_secret"]}],
               "volumes": [{"name": "model", "persistentVolumeClaim": {"claimName": cfg["pvc"]["name"]}},
                           {"name": "instructions", "configMap": {"name": CONFIG_NAME}}],
               "initContainers": init,
               "containers": [{"name": "verify", "image": HELPER_IMAGE,
                               "command": ["python3", "/instructions/prepare_worker.py"],
                               "volumeMounts": [volume_mount, {"name": "instructions", "mountPath": "/instructions", "readOnly": True}]}],
           }}}}
    return {"apiVersion": "v1", "kind": "List", "items": [pvc, configmap, job]}
