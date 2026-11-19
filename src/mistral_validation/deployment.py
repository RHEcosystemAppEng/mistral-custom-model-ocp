"""Render the packaged serving YAML for preview, installation and diagnostics."""

from .resource_templates import read_template, substitute


# Diagnostics and render-config use the image defined by the authoritative YAML.
RUNTIME_IMAGE = read_template("serving-runtime")["spec"]["containers"][0]["image"]


def render(cfg):
    runtime = read_template("serving-runtime")
    service = read_template("inference-service")
    probe_period = runtime["spec"]["containers"][0]["startupProbe"]["periodSeconds"]
    bindings = {"namespace": cfg["namespace"], "pull_secret": cfg["pull_secret"],
                "pvc_name": cfg["pvc"]["name"], "gpu_count": cfg["gpu_count"],
                "startup_failure_threshold": (cfg["serving_timeout_seconds"] + probe_period - 1) // probe_period}
    return {"apiVersion": "v1", "kind": "List", "items": [
        substitute(template, bindings) for template in (runtime, service)]}
