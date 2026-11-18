"""Remove validation workloads separately from their model storage."""

import re

from preparation import CONFIG_NAME, JOB_NAME, MODEL


# Explicit names avoid deleting unrelated workloads or namespace infrastructure.
WORKLOADS = [
    ("inferenceservice", MODEL),
    ("deployment", f"{MODEL}-predictor"),
    ("job", JOB_NAME),
    *(("job", f"mistral-2411-discover-{part}") for part in range(1, 5)),
]
RELATED = [
    ("route", MODEL),
    ("service", f"{MODEL}-predictor"),
    ("servingruntime", "mistral-runtime"),
    ("configmap", CONFIG_NAME),
    ("configmap", f"{MODEL}-kube-rbac-proxy-sar-config"),
    ("configmap", f"{MODEL}-metrics-dashboard"),
]


def cleanup(oc, cfg, command, timeout, oc_error):
    namespace = cfg["namespace"]
    claim = cfg["pvc"]["name"]
    if not re.fullmatch(r"[1-9][0-9]*s", timeout):
        print(f"FAIL {command}: CLEANUP_TIMEOUT must be a positive number of seconds")
        return 1
    scope = ("-n", namespace, f"--request-timeout={timeout}")
    print(f"{command}: namespace={namespace} PVC={claim}; "
          + ("preserving prepared weights" if command == "uninstall"
             else "explicitly deleting the claim containing model weights"), flush=True)
    target = "workloads"
    try:
        if command == "uninstall":
            for resource, name in WORKLOADS + RELATED:
                target = f"{resource}/{name}"
                removed = oc.run("delete", resource, name, *scope, "--ignore-not-found=true",
                                 "--cascade=foreground", "--wait=true", f"--timeout={timeout}", "-o", "name")
                print(f"PASS {target}: {'removed' if removed.strip() else 'already absent'}")
            target = "pods"
            pods = oc.get("pods", *scope).get("items", [])
            jobs = {name for resource, name in WORKLOADS if resource == "job"}
            for pod in pods:
                metadata = pod.get("metadata", {})
                labels = metadata.get("labels", {})
                if (labels.get("serving.kserve.io/inferenceservice") == MODEL
                        or labels.get("batch.kubernetes.io/job-name", labels.get("job-name")) in jobs
                        or any(owner.get("kind") == "Job" and owner.get("name") in jobs
                               for owner in metadata.get("ownerReferences", []))):
                    print(f"FAIL uninstall: validation pods still exist in {namespace}; "
                          "inspect pod owners and termination, then rerun make uninstall; PVC preserved")
                    return 1
            print(f"PASS uninstall: workloads removed; PVC {namespace}/{claim} and prepared weights preserved")
            return 0

        target = f"pvc/{claim}"
        try:
            oc.get("pvc", claim, *scope)
        except oc_error as error:
            if not error.not_found:
                raise
            print(f"PASS purge: PVC {namespace}/{claim} already absent")
            return 0
        for resource, name in WORKLOADS:
            target = f"{resource}/{name}"
            try:
                oc.get(resource, name, *scope)
            except oc_error as error:
                if not error.not_found:
                    raise
            else:
                print(f"FAIL purge: {target} still exists in {namespace}; run make uninstall first")
                return 1
        target = "pods"
        pods = oc.get("pods", *scope).get("items", [])
        if any(volume.get("persistentVolumeClaim", {}).get("claimName") == claim
               for pod in pods for volume in pod.get("spec", {}).get("volumes", [])):
            print(f"FAIL purge: pods still reference PVC {namespace}/{claim}; stop them and rerun make purge")
            return 1
        target = f"pvc/{claim}"
        oc.run("delete", "pvc", claim, *scope, "--ignore-not-found=true",
               "--wait=true", f"--timeout={timeout}")
        print(f"PASS purge: PVC {namespace}/{claim} removed; backing storage follows its PV reclaim policy")
        return 0
    except oc_error:
        print(f"FAIL {command}: cannot inspect or remove {target} in {namespace}; "
              "check access, events, and finalizers, then rerun. "
              + ("PVC deletion was not requested" if command == "uninstall" or target != f"pvc/{claim}"
                 else "inspect the claim to determine whether deletion completed"))
        return 1
