#!/usr/bin/env python3
"""Minimal oc fixture for operator-facing preflight tests."""
import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

args = sys.argv[1:]
scenario = os.environ.get("FAKE_OC_SCENARIO", "ready")
prepared_pvc = os.environ.get("FAKE_OC_PREPARED_PVC", "mistral-weights")
preparation_state_path = os.environ.get("FAKE_OC_PREPARATION_STATE")
preparation_state = (json.loads(Path(preparation_state_path).read_text())
                     if preparation_state_path and Path(preparation_state_path).exists() else {})

if args and args[0] in ("apply", "create", "delete", "patch", "replace", "annotate", "scale"):
    if os.environ.get("FAKE_OC_MUTATIONS"):
        with open(os.environ["FAKE_OC_MUTATIONS"], "a") as mutations:
            mutations.write(json.dumps(args) + "\n")

applied_item = None
if args and args[0] == "apply":
    applied_item = json.load(sys.stdin)
    if os.environ.get("FAKE_OC_APPLIED_RESOURCES"):
        with open(os.environ["FAKE_OC_APPLIED_RESOURCES"], "a") as applied:
            applied.write(json.dumps(applied_item) + "\n")

if scenario.startswith("cleanup"):
    state_path = Path(os.environ["FAKE_OC_STATE"])
    state = json.loads(state_path.read_text())
    verb, resource = args[:2]
    name = args[2] if len(args) > 2 and not args[2].startswith("-") else ""
    key = f"{resource}/{name}"
    if scenario == "cleanup_denied" or (scenario == "cleanup_delete_failure" and verb == "delete"):
        print("private-server-error-never-echo", file=sys.stderr)
        sys.exit(1)
    namespace = args[args.index("-n") + 1] if "-n" in args else None
    items = {key: item for key, item in state.items() if item["metadata"].get("namespace") == namespace}
    if verb == "get":
        if resource == "pods":
            print(json.dumps({"items": [item for key, item in items.items() if key.startswith("pod/")]}))
        elif key in items:
            print(json.dumps(items[key]))
        else:
            print("Error from server (NotFound)", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
    if verb == "delete":
        if key not in items:
            sys.exit(0 if "--ignore-not-found=true" in args else 1)
        if resource == "pvc" and any(volume.get("persistentVolumeClaim", {}).get("claimName") == name
                                     for item in items.values() if item.get("kind") == "Pod"
                                     for volume in item.get("spec", {}).get("volumes", [])):
            print("PVC protection: pods still reference claim", file=sys.stderr)
            sys.exit(1)
        deleted = [state.pop(key)]
        if "--cascade=foreground" in args and "--wait=true" in args:
            for owner in deleted:
                for child_key, child in list(state.items()):
                    if any(ref.get("uid") == owner["metadata"]["uid"]
                           for ref in child["metadata"].get("ownerReferences", [])):
                        deleted.append(state.pop(child_key))
        state_path.write_text(json.dumps(state))
        print(key)
        sys.exit(0)
    sys.exit(2)

if args == ["whoami"]:
    print("test-operator")
    sys.exit(0)
if args[:2] == ["auth", "can-i"]:
    print("no" if scenario == "denied" else "yes")
    sys.exit(0)
if args[:2] == ["create", "namespace"]:
    sys.exit(0)
if scenario == "preparation_success" and args and args[0] in ("apply", "delete"):
    # A background Job deletion can leave an old copier writing into staging.
    if args[0] == "delete" and "--cascade=foreground" not in args:
        print("previous preparation pod still copying", file=sys.stderr)
        sys.exit(1)
    if preparation_state_path and args[0] == "apply":
        item = applied_item
        if item["kind"] == "Job":
            if "job" in preparation_state:
                # The worker refuses to overwrite an already published model.
                item["status"] = {"conditions": [{"type": "Failed", "status": "True"}]}
                preparation_state["model_already_complete"] = True
            preparation_state["job"] = item
            if os.environ.get("FAKE_OC_RECREATE_PVC_AFTER_PREPARATION"):
                preparation_state["pvc"] = {"metadata": {"name": "mistral-weights", "uid": "replacement-uid"}}
        elif item["kind"] == "ConfigMap":
            preparation_state["configmap"] = item
        elif item["kind"] == "PersistentVolumeClaim":
            item["metadata"]["uid"] = "uid-" + item["metadata"]["name"]
            preparation_state["pvc"] = item
        Path(preparation_state_path).write_text(json.dumps(preparation_state))
    sys.exit(0)
if scenario == "inventory" and args and args[0] in ("apply", "delete"):
    sys.exit(0)
if scenario == "inventory" and args and args[0] == "logs":
    part = int(args[1].rsplit("-", 1)[1])
    print(json.loads(os.environ["FAKE_OC_INVENTORY_LOGS"])[part - 1], end="")
    sys.exit(0)
if scenario.startswith("serving_") and args and args[0] == "apply":
    sys.exit(0)
if scenario.startswith("serving_") and args and args[0] == "exec":
    print("private-tensor-name-never-echo")
    print("private-runtime-error-never-echo", file=sys.stderr)
    failed = scenario == "serving_metadata_failure"
    print("RUNTIME_CHECK " + json.dumps({"result": "failed" if failed else "passed",
                                        "stage": "tensor_mapping" if failed else "complete",
                                        "metadata_only": True, "checked_tensor_parallel_size": 1,
                                        "configured_tensor_parallel_size": int(os.environ["GPU_COUNT"]),
                                        "elapsed_seconds": 0.1}))
    sys.exit(1 if failed else 0)
if scenario.startswith("serving_") and args and args[0] == "port-forward":
    if scenario in ("serving_success", "serving_bad_inference") and not os.path.exists(os.environ["FAKE_OC_MARKER"] + "-pod-running"):
        sys.exit(1)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def do_GET(self):
            marker = os.environ["FAKE_OC_MARKER"] + "-health"
            loading = self.path == "/health" and not os.path.exists(marker) and scenario != "serving_no_transition"
            if loading:
                open(marker, "a").close()
            self.send_response(503 if loading else 200 if self.path == "/health" else 404)
            self.end_headers()

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not args[1].startswith("svc/"):
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps({"model": "wrong-model" if scenario == "serving_bad_inference" else "mistralai/mistral-large-instruct-2411",
                               "choices": [{"message": {"content": "Ready."}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    HTTPServer(("127.0.0.1", int(args[2].split(":")[0])), Handler).serve_forever()
    sys.exit(0)
if (scenario == "preparation_success" or scenario.startswith("serving_")) and args and args[0] == "logs":
    if os.environ.get("FAKE_OC_LOGS_UNAVAILABLE"):
        sys.exit(1)
    if preparation_state.get("model_already_complete"):
        print("MODEL_ALREADY_COMPLETE")
        sys.exit(0)
    if os.environ.get("FAKE_OC_BLOCK_REPORT_WRITE"):
        report_root = Path(os.environ["REPORT_DIR"])
        report_root.rmdir()
        report_root.write_text("local report destination became unavailable")
    print(os.environ.get("FAKE_OC_VERIFICATION_LOG", "PREPARATION_COMPLETE verified_files=4 peak_pvc_used_kib=1024"))
    sys.exit(0)
if not args or args[0] != "get":
    sys.exit(2)

resource = args[1]
name = args[2] if len(args) > 2 and not args[2].startswith("-") else ""
ready = {"status": {"conditions": [{"type": "Established", "status": "True"}]}}
objects = {
    "namespace": {"metadata": {"name": name}},
    "secret": {"type": "kubernetes.io/dockerconfigjson", "data": {".dockerconfigjson": "never-print-this"}},
    "storageclasses": {"items": [{"metadata": {"name": "fast", "annotations": {"storageclass.kubernetes.io/is-default-class": "true"}}, "provisioner": "csi.example.test"}]},
    "storageclass": {"metadata": {"name": name}, "provisioner": "csi.example.test"},
    "persistentvolumes": {"items": []},
    "pvc": {"metadata": {"name": name, "uid": f"uid-{name}"},
            "spec": {"storageClassName": "fast", "accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": "140Gi"}}}},
    "nodes": {"items": [{"metadata": {"name": "gpu-1"}, "spec": {"taints": [{"key": "p4-gpu", "effect": "NoSchedule"}]}, "status": {"allocatable": {"nvidia.com/gpu": "8"}, "conditions": [{"type": "Ready", "status": "True"}]}}]},
    "pods": {"items": []},
    "service": {"spec": {"ports": [{"port": 80}]}},
    "resourcequota": {"items": []},
    "crd": ready,
    "datascienceclusters.datasciencecluster.opendatahub.io": {"items": [{"spec": {"components": {"kserve": {"managementState": "Managed"}}}}]},
    "deployment": {"metadata": {"name": name, "generation": 1}, "status": {"observedGeneration": 1, "availableReplicas": 1}},
}
if scenario == "inventory":
    part = int(name.rsplit("-", 1)[1]) if resource == "job" else int(next(value for value in args if value.startswith("batch.kubernetes.io/job-name=")).rsplit("-", 1)[1])
    objects["job"] = {"status": {"conditions": [{"type": "Complete", "status": "True"}]}}
    objects["pods"] = {"items": [{"spec": {"containers": [{"image": json.loads(os.environ["FAKE_OC_WEIGHT_IMAGES"])[part - 1]}]},
                           "status": {"containerStatuses": [{"imageID": f"observed-image-{part}"}]}}]}
if scenario == "preparation_success" or scenario.startswith("serving_"):
    images = json.loads(os.environ["FAKE_OC_WEIGHT_IMAGES"])
    prepared_metadata = {"annotations": {"mistral-validation/pvc-uid": f"uid-{prepared_pvc}"}}
    prepared_volumes = [{"name": "model", "persistentVolumeClaim": {"claimName": prepared_pvc}}]
    template = {"metadata": prepared_metadata,
                "spec": {"volumes": prepared_volumes, "initContainers": [{"image": image} for image in images]}}
    if (resource == "job" and preparation_state_path and "job" not in preparation_state):
        print("Error from server (NotFound)", file=sys.stderr)
        sys.exit(1)
    if resource == "job" and os.environ.get("FAKE_OC_OBSERVER_INTERRUPTED"):
        print("local observer disconnected", file=sys.stderr)
        sys.exit(1)
    objects["job"] = preparation_state.get("job", {"spec": {"template": template}})
    objects["job"].setdefault("status", {"conditions": [{"type": "Complete", "status": "True"}]})
    if "configmap" in preparation_state:
        objects["configmap"] = preparation_state["configmap"]
    if "pvc" in preparation_state:
        objects["pvc"] = preparation_state["pvc"]
    if resource == "pods" and any(value.startswith("batch.kubernetes.io/job-name=") for value in args):
        objects["pods"] = {"items": [preparation_state.get("pod", {
            **objects["job"].get("spec", {}).get("template", {}),
            "status": {"initContainerStatuses": [
                {"imageID": f"observed-image-{index}"} for index in range(1, 5)]}})]}
if scenario.startswith("serving_"):
    if resource == "pods" and not any(value.startswith("batch.kubernetes.io/job-name=") for value in args):
        marker = os.environ["FAKE_OC_MARKER"] + "-pods"
        first = not os.path.exists(marker)
        open(marker, "a").close()
        if not first:
            open(os.environ["FAKE_OC_MARKER"] + "-pod-running", "a").close()
        objects["pods"] = {"items": [{"metadata": {"name": "mistral-predictor-abc",
                                                    "annotations": {"openshift.io/scc": "anyuid" if scenario == "serving_elevated_scc" else "restricted-v2"}},
                                       "spec": {"containers": [{"name": "kserve-container", "image": os.environ.get("FAKE_OC_RUNTIME_IMAGE", "pinned-runtime"),
                                                               "volumeMounts": [{"name": "kserve-pvc-source", "mountPath": "/mnt/models",
                                                                                 "readOnly": scenario != "serving_writable_mount"}]}],
                                                "volumes": [{"name": "kserve-pvc-source", "persistentVolumeClaim": {"claimName": prepared_pvc}}]},
                                       "status": {"phase": "Running", "conditions": [
                                           {"type": "Ready", "status": "False" if first else "True"}],
                                           "containerStatuses": [{"name": "kserve-container", "state": {"running": {
                                               "startedAt": datetime.now(timezone.utc).isoformat()}}}]}}]}
        if scenario == "serving_scc_denied":
            objects["pods"]["items"][0]["status"] = {
                "phase": "Pending", "containerStatuses": [{"state": {"waiting": {
                    "reason": "CreateContainerConfigError",
                    "message": "unable to validate against any security context constraint"}}}]}
    if resource == "inferenceservice":
        marker = os.environ["FAKE_OC_MARKER"] + "-service"
        first = not os.path.exists(marker) and scenario != "serving_pod_loading"
        open(marker, "a").close()
        objects["inferenceservice"] = {"status": {"conditions": [
            {"type": "Ready", "status": "False" if first else "True"}]}}
if scenario == "validate_ready" and resource == "inferenceservice":
    objects["inferenceservice"] = {"metadata": {"annotations": {"security.opendatahub.io/enable-auth": "true"}},
                                   "status": {"conditions": [{"type": "Ready", "status": "True"}]}}
    objects["inferenceservice"]["spec"] = {"predictor": {"model": {"resources": {"limits": {"nvidia.com/gpu": "4"}}}}}
if scenario == "validate_unready" and resource == "inferenceservice":
    objects["inferenceservice"] = {"status": {"conditions": [{"type": "Ready", "status": "False"}]}}
if scenario == "validate_no_platform_auth" and resource == "inferenceservice":
    objects["inferenceservice"] = {"status": {"conditions": [{"type": "Ready", "status": "True"}]}}
if scenario == "busy_gpu" and resource == "pods":
    objects["pods"] = {"items": [{"spec": {"nodeName": "gpu-1", "containers": [{"resources": {"limits": {"nvidia.com/gpu": "4"}}}]}, "status": {"phase": "Running"}}]}
if scenario == "tainted_gpu" and resource == "nodes":
    objects["nodes"]["items"][0]["spec"]["taints"].append({"key": "special", "effect": "NoSchedule"})
if scenario == "no_secret" and resource == "secret":
    print("Error from server (NotFound)", file=sys.stderr)
    sys.exit(1)
if (scenario == "no_pvc" or (os.environ.get("FAKE_OC_SCENARIO_PVC") == "no_pvc"
                             and "pvc" not in preparation_state)) and resource == "pvc":
    print("Error from server (NotFound)", file=sys.stderr)
    sys.exit(1)
if scenario == "static_storage" and resource == "storageclasses":
    objects["storageclasses"]["items"][0]["provisioner"] = "kubernetes.io/no-provisioner"
if scenario == "controller_down" and resource == "deployment":
    objects["deployment"]["status"]["availableReplicas"] = 0
if scenario == "no_kserve" and resource == "crd" and name == "inferenceservices.serving.kserve.io":
    print("Error from server (NotFound)", file=sys.stderr)
    sys.exit(1)
if scenario == "small_quota" and resource == "resourcequota":
    objects["resourcequota"] = {"items": [{"metadata": {"name": "team-quota"}, "status": {"hard": {"requests.nvidia.com/gpu": "6"}, "used": {"requests.nvidia.com/gpu": "2"}}}]}
if scenario == "no_namespace" and resource == "namespace":
    # A marker makes the second lookup succeed after create.
    marker = os.environ["FAKE_OC_MARKER"]
    if not os.path.exists(marker):
        print("Error from server (NotFound)", file=sys.stderr)
        open(marker, "w").close()
        sys.exit(1)
if resource not in objects:
    print("Error from server (NotFound)", file=sys.stderr)
    sys.exit(1)
print(json.dumps(objects[resource]))
