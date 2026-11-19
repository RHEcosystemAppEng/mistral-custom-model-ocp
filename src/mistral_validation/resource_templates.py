"""Load packaged YAML and bind values after parsing to preserve their types."""

from importlib.resources import files
import re

import yaml


PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")


def read_template(name):
    return yaml.safe_load(files("mistral_validation").joinpath(f"templates/{name}.yaml").read_text())


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
