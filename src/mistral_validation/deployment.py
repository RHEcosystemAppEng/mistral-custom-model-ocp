"""Render the packaged serving YAML for preview, installation and diagnostics."""

from importlib.resources import files
import re

import yaml


PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")


def read_template(name):
    return yaml.safe_load(files("mistral_validation").joinpath(f"templates/{name}.yaml").read_text())


# Diagnostics and render-config use the image defined by the authoritative YAML.
RUNTIME_IMAGE = read_template("serving-runtime")["spec"]["containers"][0]["image"]


def substitute(value, bindings):
    """Replace template values after parsing, preserving integer resource counts."""
    if isinstance(value, dict):
        return {key: substitute(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, bindings) for item in value]
    if isinstance(value, str):
        match = PLACEHOLDER.fullmatch(value)
        if match:
            return bindings[match.group(1)]
        return PLACEHOLDER.sub(lambda match: str(bindings[match.group(1)]), value)
    return value


def render(cfg):
    runtime = read_template("serving-runtime")
    service = read_template("inference-service")
    probe_period = runtime["spec"]["containers"][0]["startupProbe"]["periodSeconds"]
    bindings = {"namespace": cfg["namespace"], "pull_secret": cfg["pull_secret"],
                "pvc_name": cfg["pvc"]["name"], "gpu_count": cfg["gpu_count"],
                "startup_failure_threshold": (cfg["serving_timeout_seconds"] + probe_period - 1) // probe_period}
    return {"apiVersion": "v1", "kind": "List", "items": [
        substitute(template, bindings) for template in (runtime, service)]}
