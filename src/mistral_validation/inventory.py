"""Collect and validate the private 2411 image inventory without printing it."""

import base64
import binascii
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from .preparation import MODEL, SHA256, read_layout, render_discovery


ROOT = f"/models/{MODEL}/"
SHARDS = {f"model-{number:05d}-of-00026.safetensors" for number in range(1, 27)}


def parse_logs(logs, images):
    entries: dict[str, dict[str, Any]] = {}
    indexes = []
    counts = []
    for part, log in enumerate(logs, 1):
        count = 0
        for line in log.splitlines():
            fields = line.split("\t")
            if fields[0] == "INDEX":
                if len(fields) != 2:
                    raise ValueError("malformed inventory index output")
                try:
                    index = json.loads(base64.b64decode(fields[1], validate=True))
                    shards = set(index["weight_map"].values())
                    if not isinstance(index["weight_map"], dict) or not index["weight_map"] or not all(isinstance(name, str) for name in shards):
                        raise ValueError()
                except (ValueError, KeyError, TypeError, binascii.Error, UnicodeDecodeError) as error:
                    raise ValueError("malformed model index") from error
                indexes.append(shards)
                continue
            if len(fields) != 4 or fields[0] != "FILE":
                raise ValueError("malformed inventory output")
            _, source, size, digest = fields
            if not source.startswith(ROOT):
                raise ValueError("inventory contains a file outside the model directory")
            target = source[len(ROOT):]
            if not target or any(piece in ("", ".", "..") for piece in target.split("/")) or target.startswith(".prep") or target == ".complete.json":
                raise ValueError("inventory contains an unsafe destination")
            if any(target.startswith(other + "/") or other.startswith(target + "/") for other in entries):
                raise ValueError("inventory has conflicting file and directory destinations")
            if not re.fullmatch(r"0|[1-9][0-9]*", size) or not SHA256.fullmatch(digest):
                raise ValueError("malformed inventory file metadata")
            entry = {"image": part, "source": source, "target": target, "size": int(size), "sha256": digest}
            previous = entries.get(target)
            if previous and (previous["size"], previous["sha256"]) != (entry["size"], entry["sha256"]):
                raise ValueError("duplicate destination has conflicting content; inspect the local images")
            if not previous:
                entries[target] = entry
            count += 1
        if not count:
            raise ValueError(f"image part {part} has no regular model files")
        counts.append(count)
    if "model.safetensors.index.json" not in entries or not indexes or any(shards != SHARDS for shards in indexes):
        raise ValueError("model index must name exactly the 26 observed safetensors shards")
    missing = SHARDS - set(entries)
    if missing:
        raise ValueError(f"model index references {len(missing)} missing shards")
    return {"model": MODEL, "source_images": images, "observation": "scanned regular files in the four pinned images",
            "assembly": "overlay", "files": [entries[target] for target in sorted(entries)]}, sum(counts) - len(entries)


def difference(old, new):
    before = {entry["target"]: entry for entry in old["files"]}
    after = {entry["target"]: entry for entry in new["files"]}
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    changed = sorted(target for target in before.keys() & after.keys() if before[target] != after[target])
    fields = sorted({field for target in changed for field in before[target].keys() | after[target].keys()
                     if before[target].get(field) != after[target].get(field)})
    def ids(targets):
        return ",".join(hashlib.sha256(target.encode()).hexdigest()[:12] for target in targets[:10])
    summary = f"added={len(added)}, removed={len(removed)}, changed={len(changed)}"
    for label, targets in (("added", added), ("removed", removed), ("changed", changed)):
        if targets:
            summary += f"; {label} destination IDs={ids(targets)}"
    return summary + (f"; changed fields={','.join(fields)}" if fields else "")


def collect(oc, cfg, images, timeout, output):
    if not re.fullmatch(r"[1-9][0-9]*s", timeout):
        raise ValueError("PREPARE_TIMEOUT must be a positive number of seconds")
    namespace = cfg["namespace"]
    names = [f"mistral-2411-discover-{part}" for part in range(1, 5)]
    try:
        for name in names:
            oc.run("delete", "job", name, "-n", namespace, "--ignore-not-found=true", "--wait=true")
        oc.apply(render_discovery(cfg, images))
        deadline = time.monotonic() + int(timeout[:-1])
        logs = []
        for part, name in enumerate(names, 1):
            while True:
                job = oc.get("job", name, "-n", namespace)
                conditions = job.get("status", {}).get("conditions", [])
                if any(item.get("type") == "Complete" and item.get("status") == "True" for item in conditions):
                    break
                if any(item.get("type") == "Failed" and item.get("status") == "True" for item in conditions):
                    raise ValueError(f"inventory Job part {part} failed; inspect its pod events")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError(f"inventory Job part {part} timed out; inspect its pod events")
                time.sleep(min(5, remaining))
            pods = oc.get("pods", "-n", namespace, "-l", f"batch.kubernetes.io/job-name={name}").get("items", [])
            if len(pods) != 1 or pods[0].get("spec", {}).get("containers", [{}])[0].get("image") != images[part - 1] or not pods[0].get("status", {}).get("containerStatuses", [{}])[0].get("imageID"):
                raise ValueError(f"cannot verify pinned image for inventory Job part {part}")
            logs.append(oc.run("logs", f"job/{name}", "-n", namespace, "-c", "inventory"))
    except OSError as error:
        raise ValueError("cannot start inventory Jobs; check oc and namespace access") from error
    layout, duplicates = parse_logs(logs, images)
    read_layout_data = output.exists()
    if read_layout_data:
        old = read_layout(output, images)
        if old != layout:
            raise ValueError(f"inventory differs from {output}: {difference(old, layout)}; existing file retained")
        return len(layout["files"]), duplicates, False
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=".2411-layout-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(layout, stream, indent=2)
        stream.write("\n")
    temporary.replace(output)
    return len(layout["files"]), duplicates, True
