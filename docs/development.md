# Development

For installation and cluster use, start with the [installation guide](../README.md).
Run the following commands from the repository root.

The CLI supports Python 3.10 and newer, including Python 3.14. It uses
PyYAML to load its resource templates. Use standard `venv` and `pip`.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
mistral-validate --help
mistral-validate render-deployment
python -m mypy
python -m unittest discover -s tests -p test_preparation.py -v
make test
```

The `dev` extra installs mypy and PyYAML type stubs for typechecking; the tests use the standard-library
`unittest` runner. `make typecheck` runs the same mypy configuration. The test
suite also uses `make`, `openssl` (for local HTTPS fixtures), and `du`, available
on typical macOS/Linux development systems. Tests do not require a cluster,
`oc`, GPU libraries, or model weights. The packaging tests create an isolated
virtual environment and use pip to build and install a non-editable distribution;
pip needs access to the configured package index for build and runtime dependencies.

For a regular installation, run `python -m pip install .` instead of the editable
install above. Once installed, `mistral-validate --help` and
`mistral-validate render-deployment` work from any directory.
`python -m mistral_validation` is the equivalent module entry point, and every
Makefile workflow delegates to it using `PYTHON` (default `python3` from `PATH`).
Activate the environment before running Make, or pass
`make render-config PYTHON="$PWD/.venv/bin/python"`. Layout and report paths
are resolved relative to the directory where the command is invoked; use
absolute paths when running from elsewhere.

The discovery, preparation and serving YAML templates, preparation worker and runtime diagnostic are included in the installed
package and sent to their cluster containers when needed. GPU libraries belong
to the pinned runtime image and are not local Python dependencies.

Generated environments (`.venv/`), bytecode, typechecker caches, packaging output
(`build/`, `dist/`, `*.egg-info/`), private inventories (`.local/`), and reports
(`reports/`) are excluded from Git. Runtime and development dependencies are
declared in `pyproject.toml`.

## Templates and configuration

Resource templates are packaged under
[`src/mistral_validation/templates/`](../src/mistral_validation/templates/).
Python supplies validated configuration, inventory-derived copy commands, the
preparation worker, and the runtime diagnostic. Edit the templates to change
resource definitions; offline previews and cluster commands use the same renderers.

The public CLI remains `mistral-validate`; `python -m mistral_validation` and the
Make targets invoke the same implementation.
