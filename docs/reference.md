# Configuration and runtime reference

See the [installation guide](../README.md) for the deployment sequence.

## Preflight behavior

Preflight checks login, namespace creation or access, required namespace
permissions, the pull secret's type and presence, storage class and PVC,
namespace quota, GPU availability, RHOAI/KServe APIs, and the KServe controller.
A PVC may be absent if the selected storage class and quota allow it to be
requested during preparation. A static class also needs an Available matching
persistent volume. Existing PVCs must request at least `PVC_SIZE`, use an
appropriate access mode, and be bound. Preflight checks the pull secret's shape
without printing its data or pulling images. Each failed check prints an action
to take.

The GPU check needs permission to list nodes and pods across namespaces. It
subtracts bound pods' GPU requests from node allocatable GPUs and requires the
whole `GPU_COUNT` on one Ready, schedulable node. The planned serving pod
tolerates `p4-gpu:NoSchedule`; other blocking taints are reported. This is a
snapshot of GPU placement, not a scheduler reservation. CPU, memory, affinity,
and changes after preflight may still prevent scheduling.

## Model preparation and integrity

Preparation assembles the Mistral 2411 directory from four digest-pinned weight
images using overlay assembly. It retains the 26 indexed safetensors shards as
separate files and verifies the extracted files against a private inventory.
Image digests identify the source content; per-file sizes and SHA-256 values
verify the copied content before the completed directory is published.

Run inventory after preflight in the selected namespace. The command scans
every regular file under `/models/mistral-large-instruct-2411/` in all four
digest-pinned images and writes `.local/2411-layout.json`. The JSON records
the pinned references, that these images were scanned, and each file's source,
relative destination, byte size, and SHA-256. It checks that the model index
names exactly the observed 26 safetensors shards and that all are present.
Identical duplicate destinations are copied once; conflicting duplicates fail.
The local file is ignored by Git. Keep it private and review it before
preparation.

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
shards as separate files along with the other regular files.

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
[`src/mistral_validation/templates/`](../src/mistral_validation/templates/):
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
for deployment diagnostics. No model bytes or registry credentials
are written to it.

The 140Gi baseline is a minimum; storage requirements depend on the extracted
model and staging usage. Preparation records observed peak PVC use.
Preflight verifies the secret exists and is shaped as a registry secret; actual
registry access to all pinned images can only be checked during preparation.

## Serving behavior and runtime compatibility

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
If startup times out, inspect the predictor pod's events and logs. For an SCC
rejection, inspect the compatibility of the runtime and platform security
configuration. The templates require standard restricted or nonroot SCC access.

## Runtime diagnostic

For debugging before another full load, run:

```sh
make runtime-check NAMESPACE=mistral-custom-model-ocp
```

This requires pod exec access and a running pinned runtime container with the
prepared PVC mounted read-only; the model need not be Ready. It checks the
proposed server arguments, validates the configured tensor parallel size, then
uses the vendor model and strict loader with shape-only PyTorch meta tensors
on one isolated Gloo rank. It reads only safetensors headers, and checks tensor
mapping, shapes, and missing parameters without loading weight data. A sanitized
`runtime-check-*.json` report records the result. This diagnostic does not test
multi-GPU communication, GPU kernels, compilation, readiness, or inference;
`make install` must still pass. Cold-load time depends on storage throughput and
cache state; allow enough time in `SERVING_TIMEOUT` for the model to load.
