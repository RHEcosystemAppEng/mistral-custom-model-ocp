#!/usr/bin/env python3
"""Operator configuration and read-only cluster preflight for Mistral validation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from .preparation import JOB_NAME, PVC_UID_ANNOTATION, read_layout, render as render_preparation, render_discovery
from .inventory import collect as collect_inventory
from .deployment import RUNTIME_IMAGE, render as deployment_config
from .serving import observe as observe_serving
from .serving import verify_restart
from .validation import validate as validate_route
from .runtime_check import check as check_runtime
from .concurrency import check as check_concurrency
from .context_check import check as check_context
from .cleanup import cleanup
from .runbook import run as run_runbook, purge_run


WEIGHT_IMAGES = [
    "cp.icr.io/cp/cpd/mistral-large-instruct-2411-part1@sha256:cf6e5d18c42a9bb54784b74829ca04bd3b7c9f6db262b7b70c2fad6fa260cdbb",
    "cp.icr.io/cp/cpd/mistral-large-instruct-2411-part2@sha256:1fc2d41306d18ff2111d6d07805c06b170a2827ccd10ee5e6e10dad6a757a483",
    "cp.icr.io/cp/cpd/mistral-large-instruct-2411-part3@sha256:188ffd7990437409be9f66cf35e84ac34b87115a8edf553dca629339db15be51",
    "cp.icr.io/cp/cpd/mistral-large-instruct-2411-part4@sha256:b1465deaeb1f8beb13fa71299ab6b5751678dfcdbae64ec989a779d9052ae3ce",
]
NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
QUANTITY = re.compile(r"^(\d+)(Ki|Mi|Gi|Ti|K|M|G|T)?$")
UNITS = {None: 1, "Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
         "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4}


def quantity(value):
    match = QUANTITY.fullmatch(str(value))
    if not match:
        raise ValueError(f"invalid storage quantity: {value}")
    return int(match.group(1)) * UNITS[match.group(2)]


def config(args):
    for field in ("namespace", "pull_secret", "pvc_name"):
        value = getattr(args, field)
        if len(value) > 63 or not NAME.fullmatch(value):
            raise ValueError(f"{field.replace('_', '-')} must be a DNS label (up to 63 characters)")
    if args.storage_class and not NAME.fullmatch(args.storage_class):
        raise ValueError("storage-class must be a DNS label")
    try:
        gpu_count = int(args.gpu_count)
    except ValueError as error:
        raise ValueError("GPU_COUNT must be a positive integer") from error
    if gpu_count < 1 or str(gpu_count) != args.gpu_count:
        raise ValueError("GPU_COUNT must be a positive integer")
    size = quantity(args.pvc_size)
    if size < 140 * 1024**3:
        raise ValueError("PVC_SIZE must be at least the 140Gi baseline")
    if not re.fullmatch(r"[1-9][0-9]*s", args.serving_timeout):
        raise ValueError("SERVING_TIMEOUT must be a positive number of seconds")
    cfg = {
        "namespace": args.namespace,
        "pull_secret": args.pull_secret,
        "pvc": {"name": args.pvc_name, "size": args.pvc_size, "storage_class": args.storage_class or None},
        "gpu_count": gpu_count,
        "serving_timeout_seconds": int(args.serving_timeout[:-1]),
        "images": {"weights": WEIGHT_IMAGES, "runtime": RUNTIME_IMAGE},
    }
    predictor = deployment_config(cfg)["items"][1]["spec"]["predictor"]
    cfg["serving"] = {
        "gpu_resources": predictor["model"]["resources"],
        "tensor_parallel_size": gpu_count,
        "tolerations": predictor["tolerations"],
    }
    return cfg


class OcError(Exception):
    def __init__(self, not_found=False):
        self.not_found = not_found


class Oc:
    def __init__(self, binary):
        self.binary = binary

    def run(self, *args):
        try:
            result = subprocess.run([self.binary, *args], capture_output=True, text=True, check=False)
        except OSError as error:
            raise OcError() from error
        if result.returncode:
            # Never echo oc output: a plugin or server error might include credentials.
            raise OcError("not found" in result.stderr.lower() or "(notfound)" in result.stderr.lower())
        return result.stdout

    def get(self, *args):
        try:
            return json.loads(self.run("get", *args, "-o", "json"))
        except json.JSONDecodeError as error:
            raise OcError() from error

    def apply(self, item):
        try:
            result = subprocess.run([self.binary, "apply", "-f", "-"], input=json.dumps(item),
                                    capture_output=True, text=True, check=False)
        except OSError as error:
            raise OcError() from error
        if result.returncode:
            # The server can include object data in errors; never echo it.
            raise OcError()


def report(label, reason=None):
    print(f"{'FAIL' if reason else 'PASS'} {label}{': ' + reason if reason else ''}")
    return not reason


def check_access(oc, cfg):
    try:
        oc.run("whoami")
    except OcError:
        return report("cluster login", "log in with oc before running preflight")
    report("cluster login")
    namespace = cfg["namespace"]
    try:
        oc.get("namespace", namespace)
    except OcError as error:
        if not error.not_found:
            return report("namespace", f"cannot inspect namespace {namespace}; check cluster permissions")
        try:
            oc.run("create", "namespace", namespace)
            oc.get("namespace", namespace)
        except OcError:
            return report("namespace", f"cannot create namespace {namespace}; ask for namespace creation access")
        report("namespace", None)
        print(f"  created {namespace}")
    else:
        report("namespace")
    required = [("get", "secrets"), ("create", "jobs.batch"), ("delete", "jobs.batch"),
                ("get", "pods"), ("get", "pods/log"), ("create", "configmaps"), ("patch", "configmaps"),
                ("create", "persistentvolumeclaims"),
                ("create", "servingruntimes.serving.kserve.io"), ("create", "inferenceservices.serving.kserve.io")]
    denied = []
    for verb, resource in required:
        try:
            allowed = oc.run("auth", "can-i", verb, resource, "-n", namespace).strip().lower() == "yes"
        except OcError:
            allowed = False
        if not allowed:
            denied.append(f"{verb} {resource}")
    return report("namespace access", "grant " + ", ".join(denied) + f" in {namespace}" if denied else None)


def check_secret(oc, cfg):
    try:
        secret = oc.get("secret", cfg["pull_secret"], "-n", cfg["namespace"])
    except OcError as error:
        reason = (f"create existing registry pull secret {cfg['pull_secret']} in {cfg['namespace']}" if error.not_found
                  else f"cannot inspect {cfg['pull_secret']} in {cfg['namespace']}; grant secret read access")
        return report("IBM pull secret", reason)
    valid = secret.get("type") == "kubernetes.io/dockerconfigjson" and ".dockerconfigjson" in secret.get("data", {})
    return report("IBM pull secret", None if valid else "expected a kubernetes.io/dockerconfigjson secret with .dockerconfigjson data")


def check_storage(oc, cfg):
    requested = quantity(cfg["pvc"]["size"])
    storage_class = cfg["pvc"]["storage_class"]
    try:
        if storage_class:
            selected_class = oc.get("storageclass", storage_class)
        else:
            classes = oc.get("storageclasses").get("items", [])
            defaults = [item for item in classes if item.get("metadata", {}).get("annotations", {}).get("storageclass.kubernetes.io/is-default-class") == "true" or item.get("metadata", {}).get("annotations", {}).get("storageclass.beta.kubernetes.io/is-default-class") == "true"]
            if not defaults:
                return report("storage class", "set STORAGE_CLASS to a provisioner available on this cluster")
            selected_class = defaults[0]
            storage_class = selected_class["metadata"]["name"]
    except OcError:
        return report("storage class", "cannot read the selected storage class; check STORAGE_CLASS and permissions")
    report("storage class")
    try:
        pvc = oc.get("pvc", cfg["pvc"]["name"], "-n", cfg["namespace"])
    except OcError as error:
        if error.not_found:
            if selected_class.get("provisioner") == "kubernetes.io/no-provisioner":
                try:
                    volumes = oc.get("persistentvolumes").get("items", [])
                except OcError:
                    return report("PVC", "static storage class needs permission to list persistent volumes")
                matching = [volume for volume in volumes
                            if volume.get("spec", {}).get("storageClassName") == storage_class
                            and volume.get("status", {}).get("phase") == "Available"
                            and quantity(volume.get("spec", {}).get("capacity", {}).get("storage", "0")) >= requested
                            and any(mode in volume.get("spec", {}).get("accessModes", []) for mode in ("ReadWriteOnce", "ReadWriteMany"))]
                if not matching:
                    return report("PVC", f"no matching Available persistent volume of at least {cfg['pvc']['size']} for static class {storage_class}")
            print(f"PASS PVC: absent; {cfg['pvc']['size']} can be requested from {storage_class} during preparation")
            return True
        return report("PVC", "cannot inspect PVC; check namespace access")
    spec = pvc.get("spec", {})
    actual_class = spec.get("storageClassName") or storage_class
    if actual_class != storage_class:
        return report("PVC", f"PVC uses {actual_class}; configure STORAGE_CLASS={actual_class} or use a matching PVC")
    try:
        actual = quantity(spec.get("resources", {}).get("requests", {}).get("storage", "0"))
    except ValueError:
        actual = 0
    if actual < requested:
        return report("PVC", f"PVC requests less than {cfg['pvc']['size']}; expand it or select a larger PVC")
    if "ReadWriteOnce" not in spec.get("accessModes", []) and "ReadWriteMany" not in spec.get("accessModes", []):
        return report("PVC", "PVC needs ReadWriteOnce or ReadWriteMany access")
    phase = pvc.get("status", {}).get("phase")
    if phase and phase != "Bound":
        return report("PVC", f"PVC is {phase}; resolve provisioning before preparation")
    return report("PVC")


def check_quota(oc, cfg):
    try:
        quotas = oc.get("resourcequota", "-n", cfg["namespace"]).get("items", [])
    except OcError:
        return report("namespace quota", "need permission to list ResourceQuota in the namespace")
    try:
        oc.get("pvc", cfg["pvc"]["name"], "-n", cfg["namespace"])
        need_new_pvc = False
    except OcError as error:
        need_new_pvc = error.not_found
    deficits = []
    for quota in quotas:
        status = quota.get("status", {})
        hard, used = status.get("hard", {}), status.get("used", {})
        name = quota.get("metadata", {}).get("name", "<unnamed>")
        for key in ("requests.nvidia.com/gpu", "limits.nvidia.com/gpu"):
            if key in hard and int(hard[key]) - int(used.get(key, 0)) < cfg["gpu_count"]:
                deficits.append(f"{name} has fewer than {cfg['gpu_count']} free {key}")
        if need_new_pvc and "requests.storage" in hard:
            try:
                free = quantity(hard["requests.storage"]) - quantity(used.get("requests.storage", "0"))
            except ValueError:
                free = 0
            if free < quantity(cfg["pvc"]["size"]):
                deficits.append(f"{name} has less than {cfg['pvc']['size']} free requests.storage")
        if need_new_pvc and "persistentvolumeclaims" in hard and int(hard["persistentvolumeclaims"]) - int(used.get("persistentvolumeclaims", 0)) < 1:
            deficits.append(f"{name} has no free persistentvolumeclaims")
    return report("namespace quota", "; ".join(deficits) if deficits else None)


def pod_gpu_request(pod):
    spec = pod.get("spec", {})
    def gpu(container):
        resources = container.get("resources", {})
        return int(resources.get("requests", {}).get("nvidia.com/gpu", resources.get("limits", {}).get("nvidia.com/gpu", 0)))
    regular = sum(gpu(container) for container in spec.get("containers", []))
    init = max((gpu(container) for container in spec.get("initContainers", [])), default=0)
    return max(regular, init)


def gpu_node_issue(node):
    spec = node.get("spec", {})
    if spec.get("unschedulable"):
        return "cordoned"
    if not any(c.get("type") == "Ready" and c.get("status") == "True" for c in node.get("status", {}).get("conditions", [])):
        return "not Ready"
    for taint in spec.get("taints", []):
        if taint.get("effect") in ("NoSchedule", "NoExecute") and not (taint.get("key") == "p4-gpu" and taint.get("effect") == "NoSchedule"):
            return f"untolerated taint {taint.get('key', '<unknown>')}"
    return None


def check_gpu(oc, cfg):
    try:
        nodes = oc.get("nodes").get("items", [])
        pods = oc.get("pods", "-A").get("items", [])
    except OcError:
        return report("GPU placement", "need permission to list nodes and pods across namespaces to calculate available GPUs")
    used: dict[str, int] = {}
    for pod in pods:
        node_name = pod.get("spec", {}).get("nodeName")
        if node_name and pod.get("status", {}).get("phase") not in ("Succeeded", "Failed"):
            used[node_name] = used.get(node_name, 0) + pod_gpu_request(pod)
    candidates = []
    excluded = []
    for node in nodes:
        name = node.get("metadata", {}).get("name", "<unnamed>")
        allocatable = int(node.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", 0))
        issue = gpu_node_issue(node)
        free = max(0, allocatable - used.get(name, 0))
        if issue:
            excluded.append(f"{name}: {issue}")
        elif allocatable:
            candidates.append((name, free, allocatable))
    enough = [(name, free) for name, free, _ in candidates if free >= cfg["gpu_count"]]
    if enough:
        name, free = max(enough, key=lambda item: item[1])
        report("GPU placement")
        print(f"  {name}: {free} free nvidia.com/gpu for one {cfg['gpu_count']}-GPU serving pod")
        return True
    available = ", ".join(f"{name}: {free}/{total} free" for name, free, total in candidates) or "no schedulable GPU nodes"
    exclusions = "; ".join(excluded)
    return report("GPU placement", f"need {cfg['gpu_count']} free nvidia.com/gpu on one Ready node reachable with p4-gpu toleration; {available}" + (f"; excluded: {exclusions}" if exclusions else ""))


def check_kserve(oc, cfg):
    ok = True
    for name in ("servingruntimes.serving.kserve.io", "inferenceservices.serving.kserve.io", "datascienceclusters.datasciencecluster.opendatahub.io"):
        try:
            crd = oc.get("crd", name)
            established = any(c.get("type") == "Established" and c.get("status") == "True" for c in crd.get("status", {}).get("conditions", []))
        except OcError:
            established = False
        ok = report(name, None if established else "CRD is missing or not Established; enable RHOAI KServe or request CRD read access") and ok
    try:
        clusters = oc.get("datascienceclusters.datasciencecluster.opendatahub.io").get("items", [])
        managed = [item for item in clusters if item.get("spec", {}).get("components", {}).get("kserve", {}).get("managementState") == "Managed"]
    except OcError:
        managed = []
    if not managed:
        ok = report("RHOAI KServe", "no readable DataScienceCluster has spec.components.kserve.managementState=Managed") and ok
    elif any(item.get("status", {}).get("phase") not in (None, "Ready") for item in managed):
        ok = report("RHOAI KServe", "managed DataScienceCluster is not Ready; inspect its status") and ok
    else:
        report("RHOAI KServe")
    try:
        deployment = oc.get("deployment", "kserve-controller-manager", "-n", "redhat-ods-applications")
        status = deployment.get("status", {})
        available = status.get("availableReplicas", 0) > 0 and status.get("observedGeneration", 0) >= deployment.get("metadata", {}).get("generation", 1)
    except OcError:
        available = False
    return report("KServe controller", None if available else "kserve-controller-manager has no available replicas or cannot be inspected in redhat-ods-applications") and ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "inventory", "render-config", "render-deployment", "render-discovery", "render-preparation", "prepare", "install", "status", "validate", "restart-verify", "runtime-check", "concurrency-check", "context-check", "uninstall", "purge", "verify-run", "purge-run"))
    parser.add_argument("--oc", default="oc")
    parser.add_argument("--namespace", default="mistral-custom-model-ocp")
    parser.add_argument("--pull-secret", default="ibm-entitlement")
    parser.add_argument("--pvc-name", default="mistral-weights")
    parser.add_argument("--pvc-size", default="140Gi")
    parser.add_argument("--storage-class", default="")
    parser.add_argument("--gpu-count", default="8")
    parser.add_argument("--layout", default=".local/2411-layout.json")
    parser.add_argument("--prepare-timeout", default="14400s")
    parser.add_argument("--serving-timeout", default="3600s")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--route-url", default="")
    parser.add_argument("--cleanup-timeout", default="300s")
    parser.add_argument("--run-report", default="")
    args = parser.parse_args()
    try:
        cfg = config(args)
    except ValueError as error:
        parser.error(str(error))
    if args.command == "purge-run":
        return purge_run(Oc(args.oc), cfg, args, OcError)
    if args.command == "verify-run":
        return run_runbook(Oc(args.oc), cfg, args, OcError)
    if args.command in ("uninstall", "purge"):
        return cleanup(Oc(args.oc), cfg, args.command, args.cleanup_timeout, OcError)
    if args.command == "render-config":
        print(json.dumps(cfg, indent=2))
        return 0
    if args.command == "render-deployment":
        print(json.dumps(deployment_config(cfg), indent=2))
        return 0
    if args.command == "install":
        return install(Oc(args.oc), cfg, args.report_dir)
    if args.command == "runtime-check":
        container = deployment_config(cfg)["items"][0]["spec"]["containers"][0]
        return check_runtime(Oc(args.oc), cfg, container, args.report_dir, OcError)
    if args.command == "status":
        return serving_status(Oc(args.oc), cfg)
    if args.command == "restart-verify":
        return verify_restart(Oc(args.oc), cfg, args.report_dir, RUNTIME_IMAGE, OcError)
    if args.command in ("validate", "concurrency-check", "context-check"):
        token = os.environ.get("RHOAI_TOKEN", "")
        if not token:
            print("FAIL authentication: set RHOAI_TOKEN to a RHOAI platform model access token")
            return 1
        if args.command == "concurrency-check":
            return check_concurrency(Oc(args.oc), cfg, args.route_url, token, args.report_dir, OcError)
        if args.command == "context-check":
            return check_context(Oc(args.oc), cfg, args.route_url, token, args.report_dir, OcError)
        return validate_route(Oc(args.oc), cfg["namespace"], args.route_url, token, args.report_dir, OcError)
    if args.command == "render-discovery":
        print(json.dumps(render_discovery(cfg, WEIGHT_IMAGES), indent=2))
        return 0
    if args.command == "inventory":
        try:
            count, duplicates, created = collect_inventory(Oc(args.oc), cfg, WEIGHT_IMAGES,
                                                            args.prepare_timeout, Path(args.layout))
        except (OcError, ValueError, OSError) as error:
            reason = str(error) if isinstance(error, ValueError) else "check oc access, Job status, and the local output directory"
            print(f"FAIL inventory: {reason}")
            return 1
        print(f"PASS inventory: {count} files, {duplicates} identical duplicates; "
              f"{'saved' if created else 'unchanged'} {args.layout}")
        return 0
    if args.command in ("render-preparation", "prepare"):
        try:
            layout = read_layout(args.layout, WEIGHT_IMAGES)
            resources = render_preparation(cfg, layout, WEIGHT_IMAGES)
        except ValueError as error:
            parser.error(str(error))
        if args.command == "render-preparation":
            print(json.dumps(resources, indent=2))
            return 0
        return prepare(Oc(args.oc), cfg, resources, args.prepare_timeout, layout, args.report_dir)
    oc = Oc(args.oc)
    print(f"Preflight: namespace={cfg['namespace']} GPU_COUNT={cfg['gpu_count']} PVC={cfg['pvc']['name']} ({cfg['pvc']['size']})")
    access_ok = check_access(oc, cfg)
    checks = (check_secret, check_storage, check_quota, check_gpu, check_kserve)
    results = [check(oc, cfg) for check in checks]
    passed = access_ok and all(results)
    print("Preflight passed" if passed else "Preflight failed; resolve the FAIL checks and rerun")
    return 0 if passed else 1


def prepare(oc, cfg, resources, timeout, layout, report_dir):
    if not re.fullmatch(r"[1-9][0-9]*s", timeout):
        print("FAIL preparation: PREPARE_TIMEOUT must be a positive number of seconds")
        return 1
    inventory_bytes = sum(entry["size"] for entry in layout["files"])
    if inventory_bytes >= quantity(cfg["pvc"]["size"]):
        print("FAIL preparation: inventory exceeds PVC_SIZE; increase PVC_SIZE and rerun")
        return 1
    try:
        report_root = Path(report_dir)
        report_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=report_root):
            pass
    except OSError:
        print("FAIL preparation: REPORT_DIR is not writable; choose a writable report directory")
        return 1
    if not check_secret(oc, cfg) or not check_storage(oc, cfg):
        return 1
    namespace = cfg["namespace"]
    try:
        if verified_preparation(oc, cfg, layout):
            # Recover cluster completion after a local observer or report failure.
            # Keep the Job: its immutable template and logs are install evidence.
            identity = claim_identity(oc, cfg)
            job = oc.get("job", JOB_NAME, "-n", namespace)
            print("PASS preparation: reusing verified preparation on the same PVC")
        else:
            oc.run("delete", "job", JOB_NAME, "-n", namespace, "--ignore-not-found=true", "--cascade=foreground", "--wait=true")
            try:
                oc.get("pvc", cfg["pvc"]["name"], "-n", namespace)
            except OcError as error:
                if not error.not_found:
                    raise
                oc.apply(resources["items"][0])
            try:
                identity = claim_identity(oc, cfg)
            except OcError:
                print(f"FAIL preparation: PVC {cfg['pvc']['name']}'s name and UID are unavailable; "
                      "check PVC access and provisioning, then rerun make prepare with the same PVC_NAME")
                return 1
            # The Job's immutable pod template binds verification to this claim
            # before any preparation container starts; the claim name is in volumes.
            metadata = resources["items"][2]["spec"]["template"].setdefault("metadata", {})
            metadata.setdefault("annotations", {})[PVC_UID_ANNOTATION] = identity["uid"]
            oc.apply(resources["items"][1])
            oc.apply(resources["items"][2])
            deadline = time.monotonic() + int(timeout[:-1])
            while True:
                job = oc.get("job", JOB_NAME, "-n", namespace)
                conditions = job.get("status", {}).get("conditions", [])
                if any(item.get("type") == "Complete" and item.get("status") == "True" for item in conditions):
                    break
                if any(item.get("type") == "Failed" and item.get("status") == "True" for item in conditions):
                    raise OcError()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OcError()
                time.sleep(min(5, remaining))
    except OcError:
        print("FAIL preparation: Job did not complete. Inspect pod status and events with "
              f"oc describe job/{JOB_NAME} -n {namespace}; fix image access, LAYOUT, or PVC capacity, then rerun make prepare")
        for name in ("part-1", "part-2", "part-3", "part-4", "verify"):
            try:
                log = oc.run("logs", f"job/{JOB_NAME}", "-n", namespace, "-c", name)
            except OcError:
                continue
            for marker in ("PVC_CAPACITY_INSUFFICIENT", "COPY_FAILED", "SOURCE_FILE_MISSING", "EXTRACTION_INVENTORY_MISMATCH",
                           "EXTRACTION_SIZE_MISMATCH", "EXTRACTION_SHA256_MISMATCH", "MODEL_ALREADY_COMPLETE"):
                if marker in log:
                    print(f"  {name}: {marker}" + ("; increase PVC_SIZE or expand the PVC, then retry" if marker == "PVC_CAPACITY_INSUFFICIENT" else ""))
        return 1
    try:
        pods = oc.get("pods", "-n", namespace, "-l", f"batch.kubernetes.io/job-name={JOB_NAME}").get("items", [])
        if len(pods) != 1:
            raise OcError()
        pod = pods[0]
        actual_images = [item.get("image") for item in pod.get("spec", {}).get("initContainers", [])]
        pulled = [item.get("imageID") for item in pod.get("status", {}).get("initContainerStatuses", [])]
        if actual_images != WEIGHT_IMAGES or len(pulled) != 4 or not all(pulled):
            raise OcError()
    except OcError:
        print("FAIL preparation: cannot verify the four pinned source images from the completed Job pod")
        return 1
    try:
        if claim_identity(oc, cfg) != identity or not preparation_claim_matches(job, pod, identity):
            raise OcError()
    except OcError:
        print(f"FAIL preparation: PVC {cfg['pvc']['name']}'s name or UID changed or identity evidence is unavailable; "
              "check PVC access, stop serving, and rerun make prepare with the intended PVC_NAME; "
              "inspect any completed directory before replacing it")
        return 1
    try:
        log = oc.run("logs", f"job/{JOB_NAME}", "-n", namespace, "-c", "verify")
    except OcError:
        print("FAIL preparation: verification logs are unavailable; restore Job log access and rerun make install, "
              "or stop serving and rerun make prepare after inspecting any completed directory")
        return 1
    match = re.search(r"PREPARATION_COMPLETE verified_files=([0-9]+) peak_pvc_used_kib=([0-9]+)", log)
    if not match:
        print("FAIL preparation: Job completed without a verified completion report; inspect the Job")
        return 1
    timestamp = datetime.now(timezone.utc)
    report_path = report_root / f"preparation-{namespace}-{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    report_data = {"timestamp_utc": timestamp.isoformat(),
                   "namespace": namespace, "pvc": {**cfg["pvc"], "uid": identity["uid"]}, "job": JOB_NAME,
                   "model": layout["model"],
                   "verified_files": int(match.group(1)), "peak_pvc_used_kib": int(match.group(2)),
                   "source_images": [{"reference": reference, "observed_image_id": image_id}
                                     for reference, image_id in zip(WEIGHT_IMAGES, pulled)]}
    try:
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report_data, indent=2) + "\n")
        temporary.replace(report_path)
    except OSError:
        print("FAIL preparation: Job completed but local report could not be saved; inspect the Job and REPORT_DIR")
        return 1
    print(f"PASS preparation: {match.group(1)} files verified; observed peak PVC used {match.group(2)} KiB; report {report_path}")
    return 0


def claim_identity(oc, cfg):
    pvc = oc.get("pvc", cfg["pvc"]["name"], "-n", cfg["namespace"])
    metadata = pvc.get("metadata") or {}
    uid = metadata.get("uid")
    if (metadata.get("name") != cfg["pvc"]["name"] or not isinstance(uid, str)
            or not uid.strip() or metadata.get("deletionTimestamp")):
        raise OcError()
    return {"name": metadata["name"], "uid": uid}


def preparation_claim_matches(job, pod, identity):
    template = job.get("spec", {}).get("template", {})
    for resource in (template, pod):
        annotations = (resource.get("metadata") or {}).get("annotations") or {}
        if annotations.get(PVC_UID_ANNOTATION) != identity["uid"]:
            return False
        volumes = resource.get("spec", {}).get("volumes", [])
        if not any(volume.get("name") == "model"
                   and volume.get("persistentVolumeClaim", {}).get("claimName") == identity["name"]
                   for volume in volumes):
            return False
    return True


def verified_preparation(oc, cfg, layout=None):
    namespace = cfg["namespace"]
    try:
        job = oc.get("job", JOB_NAME, "-n", namespace)
        complete = any(condition.get("type") == "Complete" and condition.get("status") == "True"
                       for condition in job.get("status", {}).get("conditions", []))
        if not complete:
            return False
        log = oc.run("logs", f"job/{JOB_NAME}", "-n", namespace, "-c", "verify")
        completion = re.search(r"PREPARATION_COMPLETE verified_files=([1-9][0-9]*) peak_pvc_used_kib=[1-9][0-9]*", log)
        if not completion:
            return False
        if layout is not None:
            if int(completion.group(1)) != len(layout["files"]):
                return False
            volumes = job.get("spec", {}).get("template", {}).get("spec", {}).get("volumes", [])
            instructions = next((volume.get("configMap", {}).get("name") for volume in volumes
                                 if volume.get("name") == "instructions"), None)
            if not instructions:
                return False
            config = oc.get("configmap", instructions, "-n", namespace)
            try:
                prepared_layout = json.loads(config.get("data", {}).get("layout.json", ""))
            except (ValueError, TypeError):
                return False
            if prepared_layout != layout:
                return False
        pods = oc.get("pods", "-n", namespace, "-l", f"batch.kubernetes.io/job-name={JOB_NAME}").get("items", [])
        if len(pods) != 1:
            return False
        pod = pods[0]
        images = [container.get("image") for container in pod.get("spec", {}).get("initContainers", [])]
        image_ids = [container.get("imageID") for container in pod.get("status", {}).get("initContainerStatuses", [])]
        return (images == WEIGHT_IMAGES and len(image_ids) == 4 and all(image_ids)
                and preparation_claim_matches(job, pod, claim_identity(oc, cfg)))
    except OcError:
        return False


def install(oc, cfg, report_dir):
    if not verified_preparation(oc, cfg):
        print(f"FAIL install: preparation evidence must match PVC {cfg['pvc']['name']}'s current name and UID, "
              "verified inventory, and four pinned images. Check PVC access and identity; stop serving before "
              f"running make prepare PVC_NAME={cfg['pvc']['name']} NAMESPACE={cfg['namespace']} and inspect its report. "
              "For a different or recreated claim, prepare that claim; inspect any completed directory before replacing it.")
        return 1
    try:
        report_root = Path(report_dir)
        report_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=report_root):
            pass
    except OSError:
        print("FAIL install: REPORT_DIR is not writable; choose a writable report directory")
        return 1
    resources = deployment_config(cfg)["items"]
    try:
        for item in resources:
            oc.apply(item)
    except OcError:
        print("FAIL install: could not apply ServingRuntime or InferenceService; inspect namespace access and resource events")
        return 1
    print(f"PASS install: applied ServingRuntime and InferenceService in {cfg['namespace']}; observing startup")
    return observe_serving(oc, cfg, report_dir, RUNTIME_IMAGE, OcError)


def serving_status(oc, cfg):
    namespace = cfg["namespace"]
    try:
        service = oc.get("inferenceservice", "mistral-large-instruct-2411", "-n", namespace)
    except OcError:
        print(f"FAIL status: InferenceService is absent or inaccessible in {namespace}; run make install")
        return 1
    conditions = service.get("status", {}).get("conditions", [])
    ready = next((item for item in conditions if item.get("type") == "Ready"), None)
    try:
        pods = oc.get("pods", "-n", namespace,
                      "-l", "serving.kserve.io/inferenceservice=mistral-large-instruct-2411").get("items", [])
    except OcError:
        print(f"FAIL status: cannot inspect predictor pods in {namespace}; check pod read access")
        return 1
    pod_ready = any(any(condition.get("type") == "Ready" and condition.get("status") == "True"
                        for condition in pod.get("status", {}).get("conditions", [])) for pod in pods)
    if ready and ready.get("status") == "True" and pod_ready:
        print("PASS status: InferenceService and predictor pod Ready")
        return 0
    if not pod_ready:
        print(f"WAIT status: predictor pod not Ready; inspect pods and events in {namespace}")
    if not ready or ready.get("status") != "True":
        reason = ready.get("reason", "loading") if ready else "loading"
        print(f"WAIT status: InferenceService not Ready ({reason}); inspect its status in {namespace}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
