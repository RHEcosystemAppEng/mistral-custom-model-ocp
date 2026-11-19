"""Observe KServe startup and validate the vendor runtime through a local port forward."""

from datetime import datetime, timezone
from http.client import HTTPSConnection, HTTPException
import json
from pathlib import Path
import socket
import ssl
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


MODEL = "mistral-large-instruct-2411"
SERVED_MODEL = "mistralai/mistral-large-instruct-2411"
OPENER = build_opener(ProxyHandler({}))


def local_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def health(port):
    try:
        with OPENER.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except HTTPError:
        return False
    except (OSError, URLError):
        return None


CHAT_PAYLOAD = {"model": SERVED_MODEL,
                "messages": [{"role": "user", "content": "Reply with one word: ready"}],
                "max_tokens": 16, "temperature": 0}


def inference(port):
    request = Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                      data=json.dumps(CHAT_PAYLOAD).encode(), headers={"Content-Type": "application/json"})
    try:
        with OPENER.open(request, timeout=120) as response:
            if response.status != 200:
                return False
            data = json.load(response)
    except (OSError, HTTPError, URLError, ValueError):
        return False
    return valid_inference(data)


def valid_inference(data):
    if not isinstance(data, dict):
        return False
    choices = data.get("choices", [])
    return (data.get("model") == SERVED_MODEL and isinstance(choices, list) and bool(choices)
            and isinstance(choices[0], dict)
            and isinstance(choices[0].get("message"), dict)
            and isinstance(choices[0]["message"].get("content"), str)
            and bool(choices[0]["message"]["content"].strip()))


def start_forward(oc_binary, namespace, target, remote_port):
    port = local_port()
    process = subprocess.Popen([oc_binary, "port-forward", target, f"{port}:{remote_port}", "-n", namespace,
                                "--address", "127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return process, port


def model_load_seconds(pod):
    for item in pod.get("status", {}).get("containerStatuses", []):
        if item.get("name") == "kserve-container":
            started = item.get("state", {}).get("running", {}).get("startedAt")
            if started:
                try:
                    moment = datetime.fromisoformat(started.replace("Z", "+00:00"))
                    return max(0, round((datetime.now(timezone.utc) - moment).total_seconds(), 3))
                except ValueError:
                    return None
    return None


class ForwardedServiceTLS(HTTPSConnection):
    """Connect locally while verifying the certificate for the Service DNS name."""

    def __init__(self, hostname, port, context):
        super().__init__(hostname, port, context=context, timeout=120)
        self.tls_context = context

    def connect(self):
        connection = socket.create_connection(("127.0.0.1", self.port), self.timeout)
        try:
            self.sock = self.tls_context.wrap_socket(connection, server_hostname=self.host)
        except BaseException:
            connection.close()
            raise


def service_inference(oc, namespace, oc_error):
    service_name = f"{MODEL}-predictor"
    try:
        service = oc.get("service", service_name, "-n", namespace)
        ports = {item.get("port") for item in service.get("spec", {}).get("ports", [])}
        context, token = None, None
        if 8443 in ports:
            ca = oc.get("configmap", "openshift-service-ca.crt", "-n", namespace)
            context = ssl.create_default_context(cadata=ca["data"]["service-ca.crt"])
            token = oc.run("whoami", "--show-token").strip()
            if not token:
                return False
            remote_port = 8443
        elif 80 in ports:
            remote_port = 80
        else:
            return False
    except (oc_error, OSError, ValueError, KeyError):
        return False
    forward, port = start_forward(oc.binary, namespace, f"svc/{service_name}", remote_port)
    try:
        for _ in range(10):
            if forward.poll() is not None:
                return False
            if context is not None:
                connection = ForwardedServiceTLS(f"{service_name}.{namespace}.svc", port, context)
                try:
                    connection.request("POST", "/v1/chat/completions", json.dumps(CHAT_PAYLOAD),
                                       {"Content-Type": "application/json",
                                        "Authorization": f"Bearer {token}"})
                    with connection.getresponse() as response:
                        return response.status == 200 and valid_inference(json.load(response))
                except (OSError, HTTPException, ValueError):
                    pass
                finally:
                    connection.close()
            elif health(port) is True:
                return inference(port)
            time.sleep(0.5)
        return False
    finally:
        if forward.poll() is None:
            forward.terminate()
            forward.wait(timeout=5)


def mounted_readonly(pod, pvc_name, runtime_image):
    spec = pod.get("spec", {})
    containers = [item for item in spec.get("containers", []) if item.get("image") == runtime_image]
    if len(containers) != 1:
        return False
    volumes = {item.get("name") for item in spec.get("volumes", [])
               if item.get("persistentVolumeClaim", {}).get("claimName") == pvc_name}
    return any(item.get("mountPath") == "/mnt/models" and item.get("name") in volumes
               and item.get("readOnly") is True for item in containers[0].get("volumeMounts", []))


def save_report(report_dir, namespace, data, prefix="serving"):
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    path = root / f"{prefix}-{namespace}-{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    data = {"timestamp_utc": timestamp.isoformat(), "namespace": namespace, **data}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)
    return path


def is_ready(resource):
    return any(item.get("type") == "Ready" and item.get("status") == "True"
               for item in resource.get("status", {}).get("conditions", []))


def failed_health_probe(oc, namespace, pod, oc_error):
    """Use kubelet evidence when vLLM has not bound its HTTP port yet."""
    uid = pod.get("metadata", {}).get("uid")
    started = next((item.get("state", {}).get("running", {}).get("startedAt")
                    for item in pod.get("status", {}).get("containerStatuses", [])
                    if item.get("name") == "kserve-container"), None)
    if not uid or not started:
        return False
    try:
        events = oc.get("events", "-n", namespace,
                        "--field-selector", f"involvedObject.uid={uid}").get("items", [])
        start = datetime.fromisoformat(started.replace("Z", "+00:00"))
        for event in events:
            target = event.get("involvedObject", {})
            message = event.get("message", "")
            timestamp = (event.get("series", {}).get("lastObservedTime")
                         or event.get("lastTimestamp") or event.get("eventTime"))
            if (event.get("reason") == "Unhealthy" and target.get("uid") == uid
                    and target.get("fieldPath") == "spec.containers{kserve-container}"
                    and message.startswith(("Startup probe failed:", "Readiness probe failed:"))
                    and ":3000/health" in message and timestamp
                    and datetime.fromisoformat(timestamp.replace("Z", "+00:00")) >= start):
                return True
    except (oc_error, ValueError, TypeError):
        pass
    return False


def verify_restart(oc, cfg, report_dir, runtime_image, oc_error):
    """Replace the ready predictor and verify recovery from the mounted PVC."""
    namespace = cfg["namespace"]
    timeout = cfg["serving_timeout_seconds"]
    start = None
    evidence = {"model": MODEL, "pvc": cfg["pvc"]["name"], "timeout_seconds": timeout,
                "replacement_seen_seconds": None, "restart_to_ready_seconds": None,
                "observed_replacement_unready": False, "elapsed_seconds": None,
                "inference_valid": False, "result": None, "failure_detail": None}

    def finish(result, detail=None):
        evidence["result"] = result
        evidence["failure_detail"] = detail
        evidence["elapsed_seconds"] = round(time.monotonic() - start, 3) if start is not None else None
        try:
            path = save_report(report_dir, namespace, evidence, "restart")
        except OSError:
            print("FAIL restart: could not save report; check REPORT_DIR")
            return 1
        print(f"{'PASS' if result == 'passed' else 'FAIL'} restart: {detail or 'replacement pod Ready and inference succeeded'}; report {path}")
        return 0 if result == "passed" else 1

    try:
        service = oc.get("inferenceservice", MODEL, "-n", namespace)
        pods = oc.get("pods", "-n", namespace, "-l", f"serving.kserve.io/inferenceservice={MODEL}").get("items", [])
    except oc_error:
        return finish("precondition_failed", "cannot inspect existing service or predictor pod")
    ready_pods = [pod for pod in pods if is_ready(pod)]
    if not is_ready(service) or len(pods) != 1 or len(ready_pods) != 1:
        return finish("precondition_failed", "service and exactly one predictor pod must be Ready")
    original = ready_pods[0]
    identity = original.get("metadata", {})
    if not identity.get("name") or not identity.get("uid"):
        return finish("precondition_failed", "predictor pod identity is unavailable")
    if not mounted_readonly(original, cfg["pvc"]["name"], runtime_image):
        return finish("precondition_failed", "predictor pod does not mount the expected PVC read-only")
    start = time.monotonic()
    try:
        oc.run("delete", "pod", identity["name"], "-n", namespace, "--wait=false")
    except oc_error:
        return finish("delete_failed", "could not delete predictor pod")
    deadline = start + timeout
    last_detail = "replacement pod not observed"
    while time.monotonic() < deadline:
        try:
            pods = oc.get("pods", "-n", namespace, "-l", f"serving.kserve.io/inferenceservice={MODEL}").get("items", [])
            service = oc.get("inferenceservice", MODEL, "-n", namespace)
        except oc_error:
            last_detail = "cannot inspect replacement pod or service"
        else:
            original_present = any(pod.get("metadata", {}).get("uid") == identity["uid"] for pod in pods)
            replacements = [pod for pod in pods if pod.get("metadata", {}).get("uid")
                            and pod.get("metadata", {}).get("uid") != identity["uid"]]
            if replacements and evidence["replacement_seen_seconds"] is None:
                evidence["replacement_seen_seconds"] = round(time.monotonic() - start, 3)
            if len(replacements) == 1:
                replacement = replacements[0]
                if not mounted_readonly(replacement, cfg["pvc"]["name"], runtime_image):
                    return finish("mount_invalid", "replacement pod does not mount the expected PVC read-only")
                if not is_ready(replacement):
                    evidence["observed_replacement_unready"] = True
                if replacement.get("status", {}).get("phase") == "Failed":
                    return finish("pod_failed", "replacement predictor pod failed")
                if not original_present and is_ready(replacement) and is_ready(service) and time.monotonic() <= deadline:
                    evidence["restart_to_ready_seconds"] = round(time.monotonic() - start, 3)
                    if not evidence["observed_replacement_unready"]:
                        return finish("transition_not_observed", "replacement pod loading state was not observed")
                    try:
                        valid = service_inference(oc, namespace, oc_error)
                    except (OSError, subprocess.TimeoutExpired):
                        valid = False
                    if not valid:
                        return finish("inference_failed", "service chat inference failed after restart")
                    evidence["inference_valid"] = True
                    return finish("passed")
                last_detail = "replacement pod or InferenceService not Ready"
                waiting = [item.get("state", {}).get("waiting", {}).get("reason")
                           for item in replacement.get("status", {}).get("containerStatuses", [])]
                for reason in ("ImagePullBackOff", "ErrImagePull", "CrashLoopBackOff", "CreateContainerError"):
                    if reason in waiting:
                        last_detail = f"replacement container waiting: {reason}"
                        break
            elif len(replacements) > 1:
                last_detail = "multiple replacement pods observed"
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(2, remaining))
    return finish("timeout", last_detail)


def observe(oc, cfg, report_dir, runtime_image, oc_error):
    namespace = cfg["namespace"]
    deadline = time.monotonic() + cfg["serving_timeout_seconds"]
    start = time.monotonic()
    observed_unready = False
    observed_health_unready = False
    health_unready_source = None
    last_state = "waiting for predictor pod"
    forward = None
    forwarded_pod = None
    port = None

    def fail(result, state, action, scc=None):
        path = save_report(report_dir, namespace, {
            "model": MODEL, "runtime_image": runtime_image, "gpu_count": cfg["gpu_count"],
            "observed_unready": observed_unready, "observed_health_unready": observed_health_unready,
            "health_unready_source": health_unready_source,
            "health_ready": False, "inference_valid": False,
            "install_to_ready_seconds": round(time.monotonic() - start, 3),
            "model_load_seconds": None, "result": result, "last_state": state,
            "observed_scc": scc})
        print(f"FAIL serving: {action}; report {path}")
        return 1

    try:
        while True:
            try:
                pods = oc.get("pods", "-n", namespace, "-l", f"serving.kserve.io/inferenceservice={MODEL}").get("items", [])
                service = oc.get("inferenceservice", MODEL, "-n", namespace)
            except oc_error:
                pods, service = [], {}
            pod = pods[0] if pods else None
            pod_name = pod.get("metadata", {}).get("name") if pod else None
            pod_ready = any(item.get("type") == "Ready" and item.get("status") == "True"
                            for item in pod.get("status", {}).get("conditions", [])) if pod else False
            service_ready = any(item.get("type") == "Ready" and item.get("status") == "True"
                                for item in service.get("status", {}).get("conditions", []))
            if not pod_ready or not service_ready:
                observed_unready = True
            waiting = [item.get("state", {}).get("waiting", {})
                       for item in pod.get("status", {}).get("containerStatuses", [])] if pod else []
            messages = [item.get("message", "").lower() for item in waiting]
            messages += [item.get("message", "").lower() for item in service.get("status", {}).get("conditions", [])]
            if any("security context constraint" in message or "scc" in message for message in messages):
                return fail("ibm_integration_gap", "SCC rejected vendor runtime",
                            "IBM integration gap: vendor runtime was rejected by standard OpenShift SCC")
            scc = pod.get("metadata", {}).get("annotations", {}).get("openshift.io/scc") if pod else None
            if pod and scc not in ("restricted", "restricted-v2", "nonroot", "nonroot-v2"):
                return fail("ibm_integration_gap", "standard OpenShift SCC not verified",
                            "IBM integration gap: predictor pod did not use a standard OpenShift SCC", scc)
            if pod and not mounted_readonly(pod, cfg["pvc"]["name"], runtime_image):
                return fail("mount_invalid", "pinned runtime lacks read-only model PVC mount",
                            "predictor pod does not mount the completed PVC read-only at /mnt/models", scc)
            if pod_name and (pod_name != forwarded_pod or not forward or forward.poll() is not None):
                if forward and forward.poll() is None:
                    forward.terminate()
                    forward.wait(timeout=5)
                try:
                    forward, port = start_forward(oc.binary, namespace, f"pod/{pod_name}", 3000)
                    forwarded_pod = pod_name
                except OSError:
                    forward, port = None, None
            health_state = health(port) if port and forward and forward.poll() is None else None
            if health_state is False:
                observed_health_unready = True
                health_unready_source = "http_response"
            elif (not observed_health_unready and pod and not pod_ready
                  and failed_health_probe(oc, namespace, pod, oc_error)):
                observed_health_unready = True
                health_unready_source = "kubelet_probe_event"
            healthy = health_state is True
            if not (pod_ready and service_ready and healthy):
                last_state = ("pod absent" if not pod else
                              "runtime health endpoint not ready" if not healthy else
                              "pod not Ready" if not pod_ready else "InferenceService not Ready")
                failures = [item.get("reason") for item in waiting]
                if any(reason in ("ImagePullBackOff", "ErrImagePull", "CrashLoopBackOff", "CreateContainerError")
                       for reason in failures):
                    last_state = "predictor container failed: " + next(reason for reason in failures if reason)
                    break
                if pod and pod.get("status", {}).get("phase") == "Failed":
                    break
            else:
                elapsed = round(time.monotonic() - start, 3)
                duration = model_load_seconds(pod)
                valid = service_inference(oc, namespace, oc_error)
                result = ("inference_failed" if not valid else
                          "transition_not_observed" if not observed_health_unready else
                          "load_duration_unavailable" if duration is None else "passed")
                data = {"model": MODEL, "served_model": SERVED_MODEL, "runtime_image": runtime_image,
                        "gpu_count": cfg["gpu_count"], "pvc": cfg["pvc"]["name"],
                        "observed_scc": scc,
                        "observed_unready": observed_unready, "observed_health_unready": observed_health_unready,
                        "health_unready_source": health_unready_source,
                        "health_ready": True, "inference_valid": valid,
                        "install_to_ready_seconds": elapsed, "model_load_seconds": duration,
                        "result": result}
                path = save_report(report_dir, namespace, data)
                if result == "passed":
                    print(f"PASS serving: /health and Service chat inference succeeded; model loaded in {duration}s; report {path}")
                    return 0
                print(f"FAIL serving: {result}; inspect the KServe Service and rerun after a fresh deployment if loading was not observed; report {path}")
                return 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                last_state = f"timeout after {cfg['serving_timeout_seconds']}s: {last_state}"
                break
            time.sleep(min(2, remaining))
        return fail("startup_failed", last_state,
                    f"{last_state}; inspect predictor pod events and logs in {namespace}")
    except OSError:
        print("FAIL serving: could not save the serving report; check REPORT_DIR")
        return 1
    finally:
        if forward and forward.poll() is None:
            forward.terminate()
            forward.wait(timeout=5)
