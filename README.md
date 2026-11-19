# Mistral custom model on OpenShift

Bootstrapped from [mistral-validation](https://github.com/TomerG2/mistral-validation)
at commit `6cf1e67b5cbaa4a3680c4227fd9f0bfce2853b9b`.
The starting workflow retains the source's pinned IBM images and Mistral 2411
configuration. It is model-specific; arbitrary custom models are not yet supported.
The default namespace is `mistral-custom-model-ocp`.

The live cluster observations and files under `docs/evidence/` were recorded in
the source repository before this bootstrap. They describe those original runs;
this repository has not yet been validated against a live cluster.

For local development, install the Python package as described below. Run
`make test` for the tests, which use a fake OpenShift CLI, or `make render-config`
and `make render-deployment` to inspect generated objects without contacting a
cluster. Run the cluster commands below only after supplying a dedicated namespace
and its existing pull secret.

## Python setup and development

The validation CLI supports Python 3.10 and newer, including Python 3.14. It uses
PyYAML to load its resource templates. Use standard `venv` and `pip`;
neither uv nor Helm is required.

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

This repository is building an operator workflow to validate the private
`mistral-large-instruct-2411` model on Red Hat OpenShift AI (RHOAI) and KServe.
The first test target is `ai-dev04`: OpenShift 4.22, RHOAI 3.5, and 8 NVIDIA
A100 40GB GPUs. No other cluster, hardware, or GPU count is claimed as verified.

## Preflight

Install `oc` and `make`, and set up the Python environment above. Log in to the
target cluster with `oc login`.
Use a dedicated namespace. An existing namespace is selected; an absent one is
created. Supply an existing IBM `kubernetes.io/dockerconfigjson` pull secret in
that namespace through your normal secret management process. Never put the
entitlement key in this repository or in a Make variable.

```sh
make preflight NAMESPACE=mistral-custom-model-ocp PULL_SECRET=ibm-entitlement \
  GPU_COUNT=8 PVC_NAME=mistral-weights PVC_SIZE=140Gi
```

| Input | Default | Purpose |
| --- | --- | --- |
| `NAMESPACE` | `mistral-custom-model-ocp` | Dedicated validation namespace |
| `PULL_SECRET` | `ibm-entitlement` | Name of the existing namespace-scoped IBM pull secret |
| `GPU_COUNT` | `8` | GPUs for one serving pod and tensor parallel size |
| `PVC_NAME` | `mistral-weights` | Existing or future model PVC |
| `PVC_SIZE` | `140Gi` | Requested capacity; at least 140Gi |
| `STORAGE_CLASS` | cluster default | Storage class for the PVC |
| `OC` | `oc` | OpenShift CLI binary |
| `PYTHON` | `python3` | Python binary |

Preflight checks login, namespace creation or access, required namespace
permissions, the pull secret's type and presence, storage class and PVC,
namespace quota, GPU availability, RHOAI/KServe APIs, and the KServe controller.
A PVC may be absent if the selected storage class and quota allow it to be
requested during preparation. A static class also needs an Available matching
persistent volume. Existing PVCs must request at least `PVC_SIZE`, use an
appropriate access mode, and be bound. Preflight does not pull images or read
the pull secret's data. Each failed check prints an action to take.

The GPU check needs permission to list nodes and pods across namespaces. It
subtracts bound pods' GPU requests from node allocatable GPUs and requires the
whole `GPU_COUNT` on one Ready, schedulable node. The planned serving pod
tolerates `p4-gpu:NoSchedule`; other blocking taints are reported. This is a
snapshot of GPU placement, not a scheduler reservation. CPU, memory, affinity,
and changes after preflight may still prevent scheduling.

`make render-config` prints the effective JSON configuration, including the
same `GPU_COUNT` for GPU requests, limits, and tensor parallel size. It also
records the five IBM-provided digest-pinned image references from the source
[specification](https://github.com/TomerG2/mistral-validation/issues/1).
`make render-deployment` renders planned KServe ServingRuntime and
InferenceService JSON objects from those inputs, with the pinned runtime image,
pull secret name, `p4-gpu` toleration, and GPU count. The PVC root is mounted
read-only at `/mnt/models`, with the completed `mistral-large-instruct-2411`
directory beneath it. KServe's `storage.kserve.io/readonly` annotation makes
the PVC mount read-only. The runtime uses standard OpenShift security
constraints and has no privileged or fixed-UID requirement.

## Prepare the 2411 artifacts

Preparation follows the **2411-specific layout**. IBM has stated that the
`consolidated.safetensors` step does not apply to `mistral-large-instruct-2411`;
it applies to newer models such as `mistral-large-2512`. The general IBM
operator concatenates sorted `part_*` shards, but that general procedure is not
the 2411 path. Discovery against the four pinned images on `ai-dev04` observed a
26-shard `model-*-of-00026.safetensors` set and no consolidated file, which
matches IBM's statement. Preparation therefore uses `overlay` assembly.
The operator approved retaining these separate shards. This confirmed
clarification supersedes the original specification's consolidation requirement
for 2411; see the [approved scope and proposed upstream correction](docs/verification-scope.md).

IBM has no published checksum or byte size for a reconstructed file. Integrity
comes from pulling the part images by `@sha256` digest, so each shard is exactly
as built, plus this repository's own per-file size and SHA-256 verification
against the recorded inventory. IBM offers an optional extra check: compare
`ls -l` and `sha256sum` against a copy reconstructed on IBM's side. The image
digests alone do not verify extraction, which is why the inventory check exists.

Run inventory after preflight in the selected namespace. The command scans
every regular file under `/models/mistral-large-instruct-2411/` in all four
digest-pinned images and writes `.local/2411-layout.json`. The JSON records
the pinned references, that these images were scanned, and each file's source,
relative destination, byte size, and SHA-256. It checks that the model index
names exactly the observed 26 safetensors shards and that all are present.
Identical duplicate destinations are copied once; conflicting duplicates fail.
The local file is ignored by Git. Keep it private and review it before
preparation. IBM's 2411 clarification provides context; the inventory checks,
copy verification, and a successful serving run establish the result.

```sh
make preflight NAMESPACE=mistral-custom-model-ocp
make inventory NAMESPACE=mistral-custom-model-ocp
${EDITOR:-vi} .local/2411-layout.json
make prepare NAMESPACE=mistral-custom-model-ocp
make install NAMESPACE=mistral-custom-model-ocp
```

Open `.local/2411-layout.json` in a local editor to review the file list. Do
not paste its paths, sizes, or hashes into issues, command output, or tracked
files. Repeating `make inventory` with the same result leaves the file alone.
If the scanned content changes while the pinned digests stay the same, the
command reports counts and short fingerprints of added, removed, and changed
destinations, and keeps the existing file. The fingerprints are SHA-256 of
relative destination names, not file contents. Investigate the discrepancy
before moving the old file aside and rescanning. A layout generated on another cluster is accepted when
its four image references match the configured pinned digests; preparation
still verifies every copied file's size and hash.

The generated `assembly` mode is `overlay`: preparation copies the indexed
shards as separate files along with the other regular files. It does not
concatenate `part_*` files or make `consolidated.safetensors`.

```json
{
  "model": "mistral-large-instruct-2411",
  "source_images": ["<part1 pinned reference>", "<part2 pinned reference>", "<part3 pinned reference>", "<part4 pinned reference>"],
  "observation": "scanned regular files in the four pinned images",
  "assembly": "overlay",
  "files": [
    {"image": 1, "source": "/<source path>", "target": "<relative model path>",
     "size": 1, "sha256": "<64 lowercase hex characters>"}
  ]
}
```

The example shows the JSON shape; it is not an artifact inventory. Do not
commit private model content or credentials.

The authoritative resource definitions are checked in under
[`src/mistral_validation/templates/`](src/mistral_validation/templates/):
`discovery-job.yaml`, `preparation-pvc.yaml`, `preparation-configmap.yaml`,
`preparation-job.yaml`, and the repeated copy stage in
`preparation-init-container.yaml`. Python supplies validated configuration,
inventory-derived copy commands and layout JSON, and the packaged verification
worker. The four source image references remain digest-pinned in the operator
configuration. These templates require neither Helm nor cluster access to render.

Preview the four discovery Jobs before inventory, then review the preparation
PVC, ConfigMap, and Job after generating the private layout:

```sh
make -s render-discovery NAMESPACE=mistral-custom-model-ocp
umask 077
mkdir -p .local
make -s render-preparation PVC_SIZE=140Gi > .local/2411-preparation.json
make prepare PVC_SIZE=140Gi
```

`LAYOUT=/secure/path/2411-layout.json` overrides the default for either
preparation command. Both previews accept `NAMESPACE` and `PULL_SECRET`;
preparation also accepts `PVC_NAME`, `PVC_SIZE`, and `STORAGE_CLASS`. An omitted
storage class requests the cluster default. Keep the same settings and templates
between preview and execution. Inventory and preparation use these same renderers;
preparation adds the live PVC UID annotation before starting the Job. It creates
the rendered PVC only when absent and preserves the existing claim on retries.
If the layout is absent, `make prepare` fails before changing cluster resources
and tells you to run `make inventory`.

**Rendered preparation output is private:** its ConfigMap contains the complete
inventory, including paths, sizes, and hashes, and its copy commands contain file
paths. Keep it in a protected local file, out of shared logs, issues, and Git;
do not commit private inventories or credentials to templates or test fixtures.
The pull-secret field is only a reference to an existing secret.
Installed commands work offline outside the checkout as well:

```sh
mistral-validate render-discovery --namespace mistral-custom-model-ocp
mistral-validate render-preparation --layout /secure/path/2411-layout.json > /secure/path/2411-preparation.json
```

`make prepare` creates the PVC if absent, uses the existing namespace pull
secret, and checks for verified preparation on that PVC's current name and UID.
After a local interruption or report-write failure, rerunning `make prepare`
reuses the completed Job and model without recopying and saves a local preparation
report under `reports/` (or `REPORT_DIR`). The same PVC can then be used by
`make install`. Reuse requires the completed Job, verification log, and all four
pinned source images with observed image IDs; directory or marker presence alone
is insufficient. Otherwise it deletes the earlier preparation Job and starts a new one. The first
init container removes any incomplete staging directory before copying. A
failed or interrupted attempt can be retried with the same command. The final
directory appears only after every listed file has the expected size and hash;
the Job writes `.complete.json` with the pinned image references, inventory
count, and observed peak PVC contents usage.
The script samples allocated PVC contents with `du` after each copy. On a copy failure,
inspect the Job and PVC; if capacity is insufficient, expand the claim or
choose a larger `PVC_SIZE`, then retry. **140Gi is a baseline, not a fit
guarantee.** Run preparation only while serving is stopped. A successfully
completed directory is preserved; preparing a new copy requires an explicit
operator decision about replacing it.

The command also saves a timestamped JSON report under `reports/` (set
`REPORT_DIR` to change the location). It records the four requested pinned
references and the image IDs observed on the completed Job pod, plus the
verified file count, peak PVC contents usage, and the prepared PVC's name and
immutable UID (`pvc.name` and `pvc.uid`). Before starting the Job, the operator
records the current UID in its immutable pod template's
`mistral-validation/pvc-uid` annotation; the template and actual preparation
pod must both reference the selected claim and carry that UID. Preparation
checks the claim identity again before reporting success. Keep that report
with the operator's validation evidence. No model bytes or registry credentials
are written to it.

The 140Gi baseline is a minimum from the source specification, not proof that
the extracted model fits. A later preparation run must measure peak PVC use.
Preflight verifies the secret exists and is shaped as a registry secret; actual
registry access to all pinned images can only be checked during preparation.

Focused checks: run `make prepare LAYOUT=/tmp/missing-2411-layout.json` to
verify the missing-layout error; use `python3 -m unittest discover -s tests -p
test_inventory.py -v` for shard/index mismatch, duplicate conflicts, and
repeated scans. Run all local checks with
`python3 -m unittest discover -s tests -v`.

## Serve through KServe

The authoritative definitions are
[`serving-runtime.yaml`](src/mistral_validation/templates/serving-runtime.yaml) and
[`inference-service.yaml`](src/mistral_validation/templates/inference-service.yaml).
Their `{{name}}` placeholders receive validated configuration values; whole-value
GPU counts and probe thresholds render as integers. Runtime arguments, the pinned
runtime image, authentication, read-only storage, security constraints, tolerations,
and probes live in these files. Edit the templates to change those definitions.
This remains a model validation workflow; rendering and installation require no Helm.

Preview works offline, even before preparation and without `oc` installed.
After `make prepare` finishes and its report confirms the four pinned images,
review the rendered objects and install the service with the same inputs:

```sh
make -s render-deployment NAMESPACE=mistral-custom-model-ocp GPU_COUNT=8 SERVING_TIMEOUT=3600s > /tmp/mistral-serving.json
make install NAMESPACE=mistral-custom-model-ocp GPU_COUNT=8 SERVING_TIMEOUT=3600s
make status NAMESPACE=mistral-custom-model-ocp
```

The installed CLI uses the same configuration and renderer, from any directory:

```sh
mistral-validate render-deployment --namespace model-validation --pull-secret ibm-entitlement --pvc-name mistral-weights --gpu-count 8 --serving-timeout 3600s > /tmp/mistral-serving.json
mistral-validate install --namespace model-validation --pull-secret ibm-entitlement --pvc-name mistral-weights --gpu-count 8 --serving-timeout 3600s
```

The preview is a JSON Kubernetes `List` containing both resources, also valid YAML.
Install renders those same effective objects and submits them individually after
verifying preparation; it does not read the saved preview file. Keep the
configuration and installed templates the same between preview and install.
Use `PULL_SECRET` and `PVC_NAME` with Make (or `--pull-secret` and `--pvc-name`
with the CLI) to select existing resources in the effective namespace. Invalid
configuration fails before any cluster command runs.

`SERVING_TIMEOUT` is a positive number of seconds. It controls the `/health`
startup probe allowance and the operator's wait for the predictor pod and
InferenceService to become Ready. The readiness probe also uses `/health` and
keeps the pod unready while the model loads. The install command refuses to
apply serving resources unless the preparation Job completed, reported a
verified inventory, used the four expected pinned image references, and has
claim identity evidence matching the current `PVC_NAME` and its UID. Before
applying either serving resource, install reads the target PVC and compares
its identity with the preparation Job template and completed pod. A different
claim, a claim deleted and recreated under the same name, missing UID evidence
(including Jobs from older versions), or an inaccessible claim fails this
check without changing existing serving resources. A local preparation report
alone does not authorize installation; retain the completed Job and its pod
and verification logs.

On an identity failure, confirm the namespace and `PVC_NAME`, restore read
access to the claim and preparation evidence if needed, and select the claim
that was actually prepared. If using a different or recreated claim, or if the
old Job has no UID evidence, stop serving and run
`make prepare NAMESPACE=<namespace> PVC_NAME=<intended-claim>`, then retry
`make install` with the same inputs. Preparation preserves an existing
completed model directory: inspect it before deciding to replace it or use a
fresh claim. Do not relabel old evidence with a new claim's UID.

After applying the resources, install checks the runtime's `/health` endpoint
through a local pod port forward after confirming the pod's actual PVC mount
is read-only and its SCC is a standard restricted or nonroot SCC. It sends a short chat request to port
3000. Success requires a nonempty reply
whose returned model name is `mistralai/mistral-large-instruct-2411`.
For a protected RHOAI Service on port 8443, this check uses HTTPS with the
namespace's OpenShift service CA, verifies the Service DNS name, and sends the
logged-in operator's token obtained in memory from `oc whoami --show-token`.
The operator must have model access. The token is not printed or saved. Plain
HTTP Services on port 80 are also supported. External route authentication
remains a separate `make validate` check.

The pinned image is launched with `python3 -m vllm.entrypoints.openai.api_server`;
its entrypoint expects an executable before the server flags. Home and cache
directories are placed under `/var/tmp`, which is writable by the assigned
OpenShift UID. `TMPDIR`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and
`PROMETHEUS_MULTIPROC_DIR` override the image's temporary and compilation/metrics
paths; the launcher creates the metrics directory before starting vLLM.
The runtime sets `VLLM_USE_FALLBACK_LOADER=1`: the vendor image's
default sharded safetensors loader broadcasts FP8 weights through NCCL, which
fails on A100 with `FP8 reduction support begins with sm90 capable devices`.
The fallback reads the weights in each worker and avoids that broadcast. The
image's compressed-tensors implementation selects its W8A16 Marlin path on
Ampere; no model conversion or dtype override is needed.
The KV cache is explicitly kept at `auto` (the model's BF16 dtype). The vendor
loader also requires `--allowed-missing-params` for the optional attention
`q_scale`, `k_scale`, `v_scale`, and `prob_scale` parameters. Its KV-cache code
uses default scales for this mode, but the strict loader checks for missing
parameters before those defaults are assigned. The allowlist matches only
`model.layers.<number>.self_attn.attn.(q|k|v|prob)_scale`; missing model weights
and linear-layer quantization scales still fail loading.

Each install attempt that reaches startup observation saves a JSON report under
`reports/` (set `REPORT_DIR` to change it). The report records readiness,
whether `/health` was observed unready and then ready, model load duration
from the runtime container's start time, install-to-ready time, inference
validity through the KServe Service, GPU count, and the pinned runtime image.
Before vLLM binds port 3000, an unready health observation can come from a
kubelet startup/readiness probe failure for `/health` on the current pod and
container start. The report's `health_unready_source` distinguishes that
evidence from an HTTP error response. A local port-forward failure alone does
not establish the loading transition.
It contains no prompt or response text. An install attempt against an already
ready model cannot establish the loading transition and is reported as
unverified. `make status` checks the InferenceService and predictor pod conditions.
If startup times out, inspect the predictor pod's events and logs. An SCC
rejection is recorded as an IBM integration gap; it is not a verified
deployment and should not be worked around by granting elevated SCC access.

Live validation on `ai-dev04` with eight A100 GPUs passed startup, `/health`,
and chat inference through the protected KServe Service on 2026-11-17. The
observed install completed in 146.19 seconds with warm filesystem caches;
the sanitized serving report is retained locally under `reports/`.
External route authentication, restart recovery, 20 concurrent requests,
near-32K context, and final evidence still require live validation.
Cleanup is verified separately in the isolated check described below.
Complete the verified 2411 artifact inventory and preparation before install.

For faster debugging before another full load, run:

```sh
make runtime-check NAMESPACE=mistral-custom-model-ocp
```

This requires pod exec access and a running pinned runtime container with the
prepared PVC mounted read-only; the model need not be Ready. It checks the
proposed server arguments, validates the configured tensor parallel size, then
uses the vendor model and strict loader with shape-only PyTorch meta tensors
on one isolated Gloo rank. It reads only safetensors headers, and checks tensor
mapping, shapes, and missing parameters without loading weight data. On
`ai-dev04`, this reproduced the missing-attention-scales failure in 15 seconds
and passed with the narrow allowlist in 16 seconds. A sanitized
`runtime-check-*.json` report records the result. This diagnostic does not test
eight-rank communication, GPU kernels, compilation, readiness, or inference;
`make install` must still pass. The full cold load measured about 125 MiB/s on
the current gp3 PVC, so allow roughly 15–16 minutes for its weights alone.

## Verify restart recovery

With a ready InferenceService and predictor pod, run:

```sh
make restart-verify NAMESPACE=mistral-custom-model-ocp SERVING_TIMEOUT=3600s
```

The command deletes the current predictor pod and waits for a pod with a new
Kubernetes UID. It verifies that the replacement mounts the existing model PVC
read-only, observes the replacement unready while loading, then requires the
replacement pod and InferenceService to become Ready within `SERVING_TIMEOUT`.
A chat request through the KServe Service must
also succeed. This operation does not run weight preparation or reinstall the
InferenceService. It requires a single ready predictor pod before deletion.

Each attempt writes a `restart-*.json` report under `REPORT_DIR` (default
`reports/`). The report records when the replacement appeared, time from pod
deletion to readiness, inference validity, and a fixed failure detail when
recovery fails. An already Ready replacement cannot establish the loading
transition and is reported as unverified. The report contains no prompt,
response, model weights, or tokens.

## Validate the protected external route

After the InferenceService is Ready, use a RHOAI model access token for the
deployed model. Supply the token through `RHOAI_TOKEN` in the process environment;
do not put it in a Make variable, command argument, or committed file. The
rendered InferenceService requests an exposed route with platform authentication.

```sh
read -rs RHOAI_TOKEN  # paste the RHOAI model access token, then press Enter
export RHOAI_TOKEN
make validate NAMESPACE=mistral-custom-model-ocp ROUTE_URL="https://<external-route-host>"
unset RHOAI_TOKEN
```

`ROUTE_URL` may be omitted when the InferenceService reports its external HTTPS
URL in `status.url`. The command checks InferenceService readiness, sends an
unauthenticated chat request and requires HTTP 401 or 403, then sends the same
request with `Authorization: Bearer` and the platform token. Success requires
HTTP 200, the intended model ID, an assistant message, and nonempty content.
It does not compare generated wording. The validation report under `reports/`
records status codes and outcomes only; it omits the token, prompt, URL, and
response bodies. Set `REPORT_DIR` to change the report location. TLS certificate
verification remains enabled. A failed check returns a nonzero status and
identifies whether the route, authentication, readiness, or inference needs
attention.

## Verify 20 concurrent requests

After `make validate` succeeds, keep `RHOAI_TOKEN` in the environment and run:

```sh
make concurrency-check NAMESPACE=mistral-custom-model-ocp ROUTE_URL="https://<external-route-host>"
```

The command repeats the protected-route precheck, then releases 20 worker threads
at a barrier to send 20 authenticated, short chat requests to the external HTTPS
route. It requires all 20 to return HTTP 200 and a valid assistant response from
`mistralai/mistral-large-instruct-2411`. Each request has a 120-second deadline,
including connection, headers, and response body; TLS verification stays enabled
and redirects are not followed. The precheck runs separately before the timed
batch and uses the route validator's 120-second socket timeout.

This is separate from the near-32K context check: it uses short prompts and at
most 16 output tokens per request. It demonstrates concurrent client requests;
it does not measure the server scheduler's simultaneous resident sequences.

`reports/concurrency-*.json` records the deployed InferenceService model's GPU
limit as `effective_gpu_count` (requiring a matching request if specified), the
request count, batch elapsed time, deadline, and every request's ID, elapsed time,
HTTP status, and outcome. Local `GPU_COUNT` does not override the observed GPU
count. The separate `validation-*.json` records the precheck. Set `REPORT_DIR` to
choose another directory. Neither report saves credentials, URLs, prompts, or
response content. Success prints `PASS` and exits zero; any invalid response,
HTTP error, timeout, transport failure, failed prerequisite, or report write
failure exits nonzero. Per-request outcomes distinguish rejected authentication,
other HTTP errors, invalid chat/model responses, timeouts, and transport errors.
For failures, check token access, predictor readiness, route/TLS connectivity,
and model logs as appropriate. Live cluster capacity remains unverified until
this command is run against the deployment.

The resource checks follow [Kubernetes GPU resource accounting](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
and the [RHOAI 3.5 DataScienceCluster KServe configuration](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/installing_and_uninstalling_openshift_ai_self-managed/installing-and-deploying-openshift-ai_install).

## Verify the 32K context setting

After `make validate` succeeds, keep the model access token in `RHOAI_TOKEN`
and run this check separately from concurrency:

```sh
make context-check NAMESPACE=mistral-custom-model-ocp ROUTE_URL="https://<external-route-host>"
```

`ROUTE_URL` can be omitted when the InferenceService publishes its external HTTPS
URL in `status.url`. The command repeats the protected-route precheck, then uses
authenticated `POST /tokenize` requests to measure synthetic padding with the
server's chat template and generation prompt. This uses the
[vLLM tokenization API](https://docs.vllm.ai/en/v0.6.3.post1/serving/openai_compatible_server.html).
The endpoint must be accessible through the protected route and return a token
count and `max_model_len` of 32768. A missing endpoint, malformed measurement,
or different model length fails explicitly; the command does not estimate tokens
from character or word counts.

The measured input plus 16 reserved output tokens must be between 32640 and
32768 tokens, inclusive. The command searches padding lengths, then sends one
authenticated chat inference request with exactly those measured messages and
no prompt truncation. It requires HTTP 200, the intended model, a nonempty
assistant response, and usage counts matching the measured input with 1–16
generated tokens. This verifies near-limit inference, without assessing recall
or answer quality across the context. TLS verification stays enabled and
redirects are not followed. Each HTTP operation has a 120-second socket timeout;
this is not a deadline for the entire measurement and inference workflow.

`reports/context-*.json` records the deployed InferenceService's GPU limit
(requiring a matching request if present), configured and observed model lengths,
measurement method, measured input count, output reservation, returned usage,
inference elapsed time, HTTP statuses, and outcome. Local `GPU_COUNT` does not
override the observed GPU count. The separate `validation-*.json` records the
precheck. Set `REPORT_DIR` to choose another location. Reports and console output
omit credentials, URLs, prompts, token IDs, response bodies, and model content.

Success exits zero. Failed prerequisites, measurement, authentication, HTTP
rejection, transport, response validation, usage mismatch, or report writes exit
nonzero. Inspect the report's `result` and HTTP statuses to distinguish these
failures, then check endpoint access, route connectivity, predictor capacity,
and model logs as appropriate. Live near-32K capacity remains unverified until
this command succeeds against the effective deployment. Run focused local checks
with `python3 -m unittest discover -s tests -p test_context.py -v`.

## Uninstall and purge after validation

Save the validation reports, then stop the validation workloads:

```sh
make uninstall NAMESPACE=mistral-custom-model-ocp PVC_NAME=mistral-weights
```

`uninstall` removes the InferenceService and its dependent serving resources,
the ServingRuntime, preparation and inventory discovery Jobs, and preparation
ConfigMap. It waits for foreground deletion so dependent pods stop before it
reports success. The namespace, pull secret, PVC, and prepared model files are
preserved. It also removes the named predictor Deployment, Service, route, and
auth/dashboard ConfigMaps if left over from an earlier deletion.

At the end of validation, explicitly delete the model claim and its weights:

```sh
make purge NAMESPACE=mistral-custom-model-ocp PVC_NAME=mistral-weights
```

`purge` prints the target namespace and claim before deletion. It requires the
validation workloads to have been removed by `uninstall` and refuses to delete
a claim still referenced by any pod, including a terminating or completed pod.
Do not start preparation or serving concurrently with cleanup. Deletion waits
for PVC protection finalizers; it never forces deletion or removes finalizers.
Storage reclamation follows the PV's reclaim policy: `Delete` removes the
backing volume, while `Retain` requires the storage administrator to remove the
retained volume and model data. See the Kubernetes documentation on
[persistent volume reclamation](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#reclaiming).

Both commands report resources that are already absent and can be repeated.
They return nonzero for access errors, deletion timeouts, or remaining
workloads. `CLEANUP_TIMEOUT` (default `300s`) bounds each deletion and API request.
If cleanup fails, inspect the named resource's events, permissions, and
finalizers, resolve the cause, and rerun. No artifact inventory, registry access,
or model token is needed for either command. Uninstall removes the preparation
Job used as installation evidence; to validate again, purge and repeat inventory,
preparation, and installation.

For a live cleanup check, record the PVC UID before uninstall. Run `uninstall`
twice and verify that the validation controllers, their pods, route, runtime,
and preparation ConfigMap are absent while the same bound PVC remains. Then
run `purge` twice and verify that the claim is absent. With a `Delete` reclaim
policy, also wait for the corresponding PV to disappear. Save the timestamps,
namespace, claim UID, and distinct outcomes with the validation reports. Run
focused local checks with
`python3 -m unittest discover -s tests -p test_cleanup.py -v`.

Live cleanup passed on `ai-dev04` on 2026-11-18 in the temporary
`tgolan-cleanup-check` namespace, using real KServe resources, active
preparation/discovery Jobs, and a 1Gi sentinel PVC. Uninstall removed the
workloads and route while preserving the bound claim's UID and readable
sentinel file. Purge refused both running workloads and a completed pod's claim
reference, then removed the claim and its `Delete`-policy backing PV after the
pod was removed. Both commands passed on repeat, and the temporary namespace
was removed. The sanitized local evidence is
`reports/cleanup-live-20261118T134219Z.json`. This checks cleanup behavior with
sentinel data; the prepared model volume in `tgolan-mistral-validation` was
not purged by this check.

## Record one run and its limits

IBM consents to this repository being public **provided it contains no closed
artifacts**. Keep the repository private until Red Hat publication approval is
also recorded. Before any public release, review every tracked file and Git
history for closed artifacts, model content, credentials and tokens. IBM consent
does not supply Red Hat approval. Inventories and raw diagnostic files remain
private; `reports/` and `.local/` are ignored for that reason.

Use one namespace, claim, image set and configuration throughout a run. The
individual commands above are the operator runbook; this wrapper executes them
and records their outcomes in a new, private `REPORT_DIR/run-<UTC>/` directory:

```sh
# Use a fresh dedicated namespace with an existing IBM pull secret.
# Supply RHOAI_TOKEN through the environment as described above.
make verify-run NAMESPACE=mistral-verification PULL_SECRET=ibm-entitlement \
  GPU_COUNT=8 PVC_NAME=mistral-weights PVC_SIZE=140Gi \
  LAYOUT=.local/2411-layout.json SERVING_TIMEOUT=3600s
```

The command requires the target claim and InferenceService to be absent. It
runs preflight and inventory, begins preparation, interrupts the local preparation
observer once an init container is running, then reruns preparation. The second
preparation deletes the earlier Job and clears incomplete staging. This exercises
an operator interruption and retry; it does not simulate a failed storage device.
No completed model directory is replaced. Preparation can take hours, depending
on image pulls and storage. Do not run another operator command against these
resources while the wrapper is active.

The wrapper then installs, checks status and the protected external route,
verifies restart recovery, and runs concurrency and context checks separately.
It records UTC timestamps, elapsed stage times, requested parameters, observed
OCP/RHOAI versions, GPU devices/count, runtime image and observed image ID,
capacity arguments, PVC UID/capacity, preparation image IDs and peak PVC usage.
The serving stage report records readiness and model loading time. Peak PVC
usage is the preparation sampler's allocated contents maximum, not a continuous
filesystem capacity measurement. A hash of the effective ServingRuntime and
InferenceService specs is compared before and after the checks; a change fails
the run. Reports from previous runs are never automatically merged.

`verification.json` is the machine-readable index; `summary.md` is the short
reviewer handoff, alongside individual JSON check reports. The index is saved
after each stage, including failures. Unrun checks remain `not_run`; stage
failure stops subsequent checks and preserves resources for troubleshooting.
Raw child command output, API objects, environment, prompts and responses are
not copied into the bundle. If a stage fails, inspect the named stage report,
then use the troubleshooting commands below. Retain the complete private bundle.

The operator approved merging with relevance review pending, then reported a
human review with the outcome **"Mostly relevant"**. Its sample count, review UTC
timestamp, reviewer identity, and association with a recorded run are unknown.
The [review addendum](docs/verification-scope.md#human-review-addendum) records
that qualified outcome separately from historical reports. Run-specific human
review metadata remains pending; the statement alone does not make machine-readable
verification pass. HTTP success and structural chat validation do not establish
answer relevance. For a future review tied to a specific run, use three short synthetic questions
with independently known answers, inspect the responses transiently, and record
only the review UTC timestamp, sample count, a reviewer identifier approved for
disclosure, and a conclusion of relevant, mixed, or irrelevant. Do not save
questions, answers, model content, credentials, or tokens. A pending review must
remain pending in the report and in any verification claim. The wrapper exits
nonzero while review or cleanup is outstanding.

After retaining the bundle, explicitly uninstall and purge the recorded run:

```sh
make purge-run NAMESPACE=mistral-verification PVC_NAME=mistral-weights \
  RUN_REPORT=reports/run-<UTC>/verification.json
unset RHOAI_TOKEN
```

This runs uninstall twice, then purge twice, and records each result and elapsed
time in the same bundle. It checks the namespace, claim name and recorded PVC
UID before removing resources. `purge-run` returns zero when cleanup succeeds;
the report's overall verification remains incomplete with human review pending.
The existing `make uninstall` and `make purge` commands remain available for
recovery after an early workflow failure before the PVC UID could be recorded.
With a `Retain` policy, arrange deletion of retained model data with the storage
administrator and record its completion privately. A PVC deletion alone does
not establish removal of retained data.

Troubleshooting and recovery:

| Symptom | Operator action |
| --- | --- |
| Preflight fails | Resolve its permissions, secret, quota, storage or GPU placement failure; GPUs already allocated to another service are unavailable. |
| Image pull fails | Inspect `oc describe pod -n <namespace> <pod>` locally; verify entitlement covers all five pinned images and the secret is in this namespace. Never print secret data. |
| Preparation fails | Stop serving, inspect Job events and fixed failure markers, confirm capacity and inventory, then rerun `make prepare` with identical inputs. A completed directory is deliberately preserved. |
| Loading or readiness fails | Use `make status`, inspect predictor events and logs locally, and the runtime notes above. Allow the documented cold-load time; use `make runtime-check` for the loader diagnostic. |
| Route authentication fails | Check external HTTPS URL, platform authentication and token permissions/expiry, then rerun `make validate`. Keep TLS verification enabled. |
| Recovery or capacity check fails | Inspect its JSON result, predictor readiness and local logs; rerun the individual command after resolving the cause. Do not relabel the failed bundle as passed. |
| Cleanup fails | Resolve resource access/finalizer or remaining-pod issues and repeat explicit cleanup. Never force finalizers away. |

The complete automated workflow passed on `ai-dev04` on 2026-11-18, including
preparation interruption/retry, readiness, protected-route inference, restart,
20 concurrent requests, the separate near-32K check, uninstall and explicit PVC
and backing-PV removal. The tested configuration was OCP 4.22.20, RHOAI 3.5.2,
and eight NVIDIA A100-SXM4-40GB GPUs. See the
[short verification summary](docs/evidence/ai-dev04-20261118.md) and
[machine-readable bundle](docs/evidence/ai-dev04-20261118.json).
That historical bundle records human answer relevance as pending. The later
operator outcome, **"Mostly relevant"**, is documented in the
[separate addendum](docs/verification-scope.md#human-review-addendum) without
assigning it to that run or changing its machine-readable result. These are
source-repository observations, not a new live validation of this checkout. Any claim
must name these exact versions, GPU type/count, pinned images and configuration
hash, and disclose pending or failed checks. Other configurations and answer
quality remain outside this claim.
