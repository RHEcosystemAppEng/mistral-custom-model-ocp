#!/usr/bin/env python3
"""Verify a copied inventory and atomically publish the model directory."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(os.environ.get("PREPARATION_ROOT", "/pvc"))
STAGE = ROOT / ".mistral-2411-staging"
FINAL = ROOT / "mistral-large-instruct-2411"
LAYOUT_PATH = Path(os.environ.get("PREPARATION_LAYOUT", "/instructions/layout.json"))


def peak_kib():
    observed = int((ROOT / ".mistral-2411-peak-kib").read_text())
    current = int(subprocess.check_output(["du", "-sk", str(ROOT)], text=True).split()[0])
    return max(observed, current)


def verify():
    layout = json.loads(LAYOUT_PATH.read_text())
    expected = {entry["target"]: entry for entry in layout["files"]}
    actual = {str(path.relative_to(STAGE)) for path in STAGE.rglob("*") if path.is_file()}
    if actual != set(expected):
        raise ValueError("EXTRACTION_INVENTORY_MISMATCH: inspect pinned images and LAYOUT")
    for target, entry in expected.items():
        path = STAGE / target
        if path.is_symlink() or path.stat().st_size != entry["size"]:
            raise ValueError("EXTRACTION_SIZE_MISMATCH: inspect pinned images and LAYOUT")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError("EXTRACTION_SHA256_MISMATCH: inspect pinned images and LAYOUT")
    peak = peak_kib()
    evidence = {"model": layout["model"], "source_images": layout["source_images"],
                "verified_files": len(expected), "peak_pvc_used_kib": peak,
                "verification": "extracted files matched the supplied per-file size and SHA-256 inventory"}
    (STAGE / ".complete.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if FINAL.exists():
        raise ValueError("MODEL_ALREADY_COMPLETE: stop serving and inspect the completed directory before preparing again")
    STAGE.rename(FINAL)
    print(f"PREPARATION_COMPLETE verified_files={len(expected)} peak_pvc_used_kib={peak}")


if __name__ == "__main__":
    try:
        verify()
    except (OSError, ValueError) as error:
        # Avoid printing paths or file contents from private model artifacts.
        message = str(error)
        if not message.startswith(("EXTRACTION_", "MODEL_ALREADY_COMPLETE")):
            message = "PREPARATION_FAILED: inspect Job status and PVC capacity; increase PVC_SIZE if full"
        print(message, file=sys.stderr)
        sys.exit(1)
