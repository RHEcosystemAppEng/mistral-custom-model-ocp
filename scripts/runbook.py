"""Run the operator workflow in a fresh namespace/claim and retain safe evidence."""

from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from preparation import JOB_NAME
from serving import MODEL


STAGES = ("preflight", "inventory", "preparation_interruption", "prepare", "install", "status",
          "validate", "restart-verify", "concurrency-check", "context-check", "human_review",
          "uninstall", "uninstall_repeat", "purge", "purge_repeat")


def now():
    return datetime.now(timezone.utc).isoformat()


def snapshot(oc, cfg):
    """Select fields deliberately: never persist full API objects or environment."""
    namespace = cfg["namespace"]
    service = oc.get("inferenceservice", MODEL, "-n", namespace)
    runtime = oc.get("servingruntime", "mistral-runtime", "-n", namespace)
    pods = oc.get("pods", "-n", namespace, "-l", f"serving.kserve.io/inferenceservice={MODEL}")["items"]
    active = [pod for pod in pods if not pod.get("metadata", {}).get("deletionTimestamp")]
    if len(active) != 1:
        raise ValueError()
    pod = active[0]
    container = next(c for c in pod["spec"]["containers"] if c["name"] == "kserve-container")
    status = next(c for c in pod["status"]["containerStatuses"] if c["name"] == "kserve-container")
    pvc = oc.get("pvc", cfg["pvc"]["name"], "-n", namespace)
    cluster = oc.get("clusterversion", "version")
    versions = cluster["status"]["desired"]["version"]
    operators = oc.get("csv", "-n", "redhat-ods-operator")["items"]
    rhoai = sorted({c["spec"]["version"] for c in operators
                    if c["metadata"]["name"].startswith("rhods-operator.")})
    if len(rhoai) != 1:
        raise ValueError()
    devices = oc.run("exec", pod["metadata"]["name"], "-n", namespace, "-c", "kserve-container", "--",
                     "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits")
    gpus = []
    for line in devices.strip().splitlines():
        name, memory = line.rsplit(",", 1)
        gpus.append({"type": name.strip(), "memory_mib": int(memory.strip())})
    if len(gpus) != cfg["gpu_count"]:
        raise ValueError()
    # Hash all effective runtime/predictor inputs without retaining arbitrary env values.
    fingerprint = hashlib.sha256(json.dumps(
        [runtime["spec"], service["spec"]], sort_keys=True).encode()).hexdigest()
    return {"timestamp_utc": now(), "cluster_id": cluster["spec"]["clusterID"], "ocp_version": versions, "rhoai_version": rhoai[0],
            "inferenceservice_uid": service["metadata"]["uid"],
            "configuration_sha256": fingerprint, "pvc_uid": pvc["metadata"]["uid"],
            "pvc_capacity": pvc["status"]["capacity"],
            "gpu_devices": gpus, "actual_gpu_count": len(gpus),
            "gpu_resources": container["resources"],
            "capacity_arguments": [a for a in container.get("args", []) if a.startswith(
                ("--max-model-len=", "--max-num-seqs=", "--tensor-parallel-size="))],
            "runtime_image": container["image"], "runtime_image_id": status["imageID"]}


def write_summary(root, evidence):
    observed = evidence.get("tested_configuration", {})
    lines = [f"Verification: {evidence['result']}", f"Started: {evidence['started_utc']}",
             f"Updated: {evidence['updated_utc']}", f"Namespace: {evidence['namespace']}"]
    if observed:
        lines += [f"OCP {observed['ocp_version']}; RHOAI {observed['rhoai_version']}; "
                  f"{observed['actual_gpu_count']} GPUs: " + ", ".join(sorted({g['type'] for g in observed['gpu_devices']})),
                  f"Configuration SHA-256: {observed['configuration_sha256']}"]
    lines += [f"{stage}: {value['result']}" for stage, value in evidence["checks"].items()]
    lines += ["Human relevance conclusion: " + evidence["human_review"]["conclusion"],
              "Scope: only the recorded configuration; no other hardware or deployment is verified."]
    (root / "summary.md").write_text("\n\n".join(lines) + "\n")


def persist(root, evidence):
    evidence["updated_utc"] = now()
    temporary = root / "verification.tmp"
    temporary.write_text(json.dumps(evidence, indent=2) + "\n")
    temporary.replace(root / "verification.json")
    write_summary(root, evidence)


def purge_run(oc, cfg, args, oc_error):
    """Explicitly finish cleanup of a recorded run; retain pending human review."""
    from cleanup import cleanup

    root = Path(args.run_report).resolve().parent
    try:
        if not re.fullmatch(r"[1-9][0-9]*s", args.cleanup_timeout):
            raise ValueError()
        evidence = json.loads(Path(args.run_report).read_text())
        if evidence["namespace"] != cfg["namespace"] or evidence["requested_configuration"]["pvc"]["name"] != cfg["pvc"]["name"]:
            raise ValueError()
        # A repeated cleanup must not remove a replacement service either.
        try:
            service = oc.get("inferenceservice", MODEL, "-n", cfg["namespace"])
            expected_service = evidence.get("tested_configuration", {}).get("inferenceservice_uid")
            if not expected_service or service["metadata"]["uid"] != expected_service:
                raise ValueError()
        except oc_error as error:
            if not error.not_found:
                raise
        # Never delete a different PVC recreated under the same name.
        claim_present = False
        volume = evidence.get("backing_volume", {}).get("name")
        expected_uid = evidence.get("tested_configuration", {}).get("pvc_uid")
        try:
            claim = oc.get("pvc", cfg["pvc"]["name"], "-n", cfg["namespace"])
            claim_present = True
            if not expected_uid or claim["metadata"]["uid"] != expected_uid:
                raise ValueError()
            volume = claim["spec"].get("volumeName")
            pv = oc.get("pv", volume) if volume else {}
            evidence["storage_reclaim_policy"] = pv.get("spec", {}).get("persistentVolumeReclaimPolicy")
            evidence["backing_volume"] = {"name": volume, "uid": pv.get("metadata", {}).get("uid")}
            persist(root, evidence)
        except oc_error as error:
            if not error.not_found:
                raise
        for name, command in (("uninstall", "uninstall"), ("uninstall_repeat", "uninstall"),
                              ("purge", "purge"), ("purge_repeat", "purge")):
            started = time.monotonic()
            value = {"started_utc": now(), "result": "running"}
            evidence["checks"][name] = value
            persist(root, evidence)
            code = cleanup(oc, cfg, command, args.cleanup_timeout, oc_error)
            value.update(result="passed" if code == 0 else "failed", finished_utc=now(),
                         elapsed_seconds=round(time.monotonic() - started, 3))
            persist(root, evidence)
            if code:
                return 1
            if name == "uninstall_repeat" and claim_present:
                preserved = oc.get("pvc", cfg["pvc"]["name"], "-n", cfg["namespace"])
                if preserved["metadata"]["uid"] != expected_uid or preserved["status"]["phase"] != "Bound":
                    raise ValueError()
                evidence["cleanup_claim_preserved"] = True
        if volume and evidence.get("storage_reclaim_policy") == "Delete":
            deadline = time.monotonic() + int(args.cleanup_timeout[:-1])
            while True:
                try:
                    remaining = oc.get("pv", volume)
                    if remaining["metadata"]["uid"] != evidence["backing_volume"]["uid"]:
                        raise ValueError()
                except oc_error as error:
                    if error.not_found:
                        evidence["backing_volume_removed"] = True
                        break
                    raise
                if time.monotonic() >= deadline:
                    evidence["backing_volume_removed"] = False
                    persist(root, evidence)
                    return 1
                time.sleep(2)
        if evidence["result"] == "automated_checks_passed_human_review_pending":
            evidence["result"] = "incomplete_human_review_pending"
        persist(root, evidence)
        return 0
    except (OSError, ValueError, KeyError, TypeError, oc_error):
        print("FAIL purge-run: report/namespace/PVC identity or cluster access could not be verified")
        return 1


def run(oc, cfg, args, oc_error):
    root = Path(args.report_dir) / ("run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    evidence: dict[str, Any] = {"schema_version": 1, "started_utc": now(),
        "namespace": cfg["namespace"], "requested_configuration": cfg,
        "checks": {stage: {"result": "not_run"} for stage in STAGES},
        "human_review": {"result": "pending", "sample_count": 0, "conclusion": "pending"},
        "result": "incomplete"}
    root.mkdir(parents=True)
    os.chmod(root, 0o700)

    def save():
        persist(root, evidence)

    common = [sys.executable, str(Path(__file__).with_name("operator.py"))]
    options = ["--oc", args.oc, "--namespace", cfg["namespace"], "--pull-secret", cfg["pull_secret"],
               "--pvc-name", cfg["pvc"]["name"], "--pvc-size", cfg["pvc"]["size"],
               "--storage-class", cfg["pvc"]["storage_class"] or "", "--gpu-count", str(cfg["gpu_count"]),
               "--layout", args.layout, "--report-dir", str(root), "--prepare-timeout", args.prepare_timeout,
               "--serving-timeout", args.serving_timeout, "--cleanup-timeout", args.cleanup_timeout,
               "--route-url", args.route_url]

    def stage(name, command=None):
        started = time.monotonic()
        value = {"started_utc": now(), "result": "running"}
        evidence["checks"][name] = value
        save()
        code = subprocess.run(common + [command or name] + options,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        value.update(result="passed" if code == 0 else "failed", finished_utc=now(),
                     elapsed_seconds=round(time.monotonic() - started, 3))
        save()
        print(f"{name}: {value['result']}", flush=True)
        return code == 0

    save()
    print(f"Run evidence: {root}", flush=True)
    try:
        if not os.environ.get("RHOAI_TOKEN"):
            evidence["failure"] = "RHOAI_TOKEN is required before starting"
            return 1
        # A fresh claim avoids overwriting prepared weights or disrupting a service.
        for resource, name in (("pvc", cfg["pvc"]["name"]), ("inferenceservice", MODEL)):
            try:
                oc.get(resource, name, "-n", cfg["namespace"])
            except oc_error as error:
                if not error.not_found:
                    raise
            else:
                evidence["failure"] = "use a fresh dedicated namespace and absent claim/service"
                return 1
        for name in ("preflight", "inventory"):
            if not stage(name):
                return 1
        # Exercise retry after interruption, before any completed directory exists.
        started = time.monotonic()
        evidence["checks"]["preparation_interruption"] = {"started_utc": now(), "result": "running"}
        save()
        process = subprocess.Popen(common + ["prepare"] + options,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise ValueError()
                try:
                    pods = oc.get("pods", "-n", cfg["namespace"], "-l",
                                  f"batch.kubernetes.io/job-name={JOB_NAME}").get("items", [])
                    if any(any("running" in s.get("state", {}) for s in
                               p.get("status", {}).get("initContainerStatuses", [])) for p in pods):
                        break
                except oc_error:
                    pass
                time.sleep(2)
            else:
                raise ValueError()
            process.terminate()
            process.wait(timeout=30)
            # prepare's next invocation deletes the interrupted Job and staging directory.
            evidence["checks"]["preparation_interruption"].update(
                result="passed", finished_utc=now(), elapsed_seconds=round(time.monotonic() - started, 3))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        save()
        for name in ("prepare", "install", "status"):
            if not stage(name):
                return 1
        preparation_reports = list(root.glob("preparation-*.json"))
        if len(preparation_reports) != 1:
            raise ValueError()
        preparation = json.loads(preparation_reports[0].read_text())
        evidence["preparation"] = {key: preparation[key] for key in (
            "timestamp_utc", "verified_files", "peak_pvc_used_kib", "source_images")}
        evidence["tested_configuration"] = snapshot(oc, cfg)
        save()
        for name in ("validate", "restart-verify", "concurrency-check", "context-check"):
            if not stage(name):
                return 1
        after = snapshot(oc, cfg)
        before = evidence["tested_configuration"]
        if any(before[key] != after[key] for key in (
                "configuration_sha256", "inferenceservice_uid", "pvc_uid", "gpu_devices", "runtime_image_id")):
            evidence["failure"] = "configuration changed during verification"
            return 1
        evidence["final_configuration"] = after
        # Human review is intentionally never inferred from HTTP or automated success.
        evidence["checks"]["human_review"] = {"result": "pending"}
        evidence["result"] = "automated_checks_passed_human_review_pending"
        return 1
    except (oc_error, OSError, ValueError, KeyError, TypeError, StopIteration, subprocess.SubprocessError):
        evidence["failure"] = "workflow could not finish; inspect cluster status and stage reports"
        return 1
    finally:
        for value in evidence["checks"].values():
            if value["result"] == "running":
                value.update(result="aborted", finished_utc=now())
        evidence["finished_utc"] = now()
        save()
        print(f"Evidence saved in {root}; follow the runbook for human review, uninstall and explicit purge.")
