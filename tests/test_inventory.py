"""Operator checks for generating and reusing the private 2411 inventory."""

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_preparation import FAKE_OC, IMAGES, ROOT


MODEL_ROOT = "/models/mistral-large-instruct-2411/"


def logs(changed=False, missing=False, conflict=False, malformed=False, no_index=False):
    index = {"weight_map": {f"layer-{number}": f"model-{number:05d}-of-00026.safetensors"
                            for number in range(1, 27)}}
    result = [[] for _ in range(4)]
    for number in range(1, 27):
        name = f"model-{number:05d}-of-00026.safetensors"
        if missing and number == 26:
            continue
        digest = hashlib.sha256(name.encode()).hexdigest()
        if changed and number == 1:
            digest = "b" * 64
        result[0].append(f"FILE\t{MODEL_ROOT}{name}\t{number}\t{digest}")
    for part in range(1, 4):
        result[part].append(f"FILE\t{MODEL_ROOT}config-{part}.json\t0\t{'0' * 64}")
    result[3].append("FILE\t" + MODEL_ROOT + "model.safetensors.index.json\t1\t" + "a" * 64)
    if not no_index:
        result[3].append("INDEX\t" + ("not-base64" if malformed else base64.b64encode(json.dumps(index).encode()).decode()))
    if conflict:
        result[1].append(f"FILE\t{MODEL_ROOT}model-00001-of-00026.safetensors\t999\t{'f' * 64}")
    else:
        result[1].append(result[0][0])
    return ["\n".join(lines) + "\n" for lines in result]


class InventoryCommandTests(unittest.TestCase):
    def run_inventory(self, path, output):
        env = {**os.environ, "OC": str(FAKE_OC), "LAYOUT": str(path),
               "FAKE_OC_SCENARIO": "inventory", "FAKE_OC_WEIGHT_IMAGES": json.dumps(IMAGES),
               "FAKE_OC_INVENTORY_LOGS": json.dumps(output)}
        return subprocess.run(["make", "--no-print-directory", "inventory"], cwd=ROOT,
                              env=env, capture_output=True, text=True)

    def test_inventory_writes_layout_and_identical_rescan_leaves_it_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2411-layout.json"
            first = self.run_inventory(path, logs())
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            layout = json.loads(path.read_text())
            self.assertEqual(len(layout["files"]), 30)
            self.assertEqual(layout["source_images"], IMAGES)
            self.assertEqual(layout["observation"], "scanned regular files in the four pinned images")
            self.assertNotIn("confirmation", layout)
            before = path.stat().st_mtime_ns
            second = self.run_inventory(path, logs())
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(path.stat().st_mtime_ns, before)
            self.assertIn("unchanged", second.stdout)
            self.assertNotIn(MODEL_ROOT, second.stdout)

    def test_changed_rescan_fails_and_preserves_existing_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2411-layout.json"
            self.assertEqual(self.run_inventory(path, logs()).returncode, 0)
            original = path.read_bytes()
            result = self.run_inventory(path, logs(changed=True))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("changed=1", result.stdout)
            self.assertIn("changed destination IDs=", result.stdout)
            self.assertEqual(path.read_bytes(), original)

    def test_missing_shard_conflict_and_malformed_index_fail_without_output(self):
        for output, reason in ((logs(missing=True), "missing shards"),
                               (logs(conflict=True), "conflicting content"),
                               (logs(malformed=True), "malformed model index"),
                               (logs(no_index=True), "model index must name")):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "2411-layout.json"
                result = self.run_inventory(path, output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(reason, result.stdout)
                self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
