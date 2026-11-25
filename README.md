# Mistral custom model on OpenShift

Install and serve **Mistral Large Instruct 2411** on Red Hat OpenShift AI (RHOAI)
with KServe. The installer discovers the model files in IBM-provided images,
prepares a verified model volume, and deploys an authenticated inference service.

The current implementation uses digest-pinned images and runtime settings for
`mistral-large-instruct-2411`. Supporting another model requires changes to the
image configuration, artifact preparation, and runtime settings.

## Prerequisites

- An OpenShift AI cluster with KServe enabled and its controller available in
  `redhat-ods-applications`.
- NVIDIA GPUs exposed as `nvidia.com/gpu`. The default configuration requests
  eight GPUs on one node and uses tensor parallel size eight. The runtime includes
  A100/Ampere loader settings; changing GPU count or hardware requires checking
  model fit and runtime compatibility.
- Persistent storage with at least `140Gi` requested capacity. This is a minimum,
  not a fit guarantee; allow capacity for preparation staging and the model.
- Entitlement to pull all four IBM weight images and the pinned runtime image,
  plus an existing namespace-scoped `kubernetes.io/dockerconfigjson` pull secret.
- Python 3.10 or newer, `oc`, and `make`. Log in with `oc login` and use a dedicated
  namespace with permissions to manage the installation resources. Preflight also
  needs permission to inspect cluster APIs, storage classes, nodes, and pods
  across namespaces.

Preflight checks the platform APIs and available resources. Platform versions,
hardware, storage performance, and model capacity must be checked for your own
deployment; preflight does not guarantee that the model will load or fit.
The templates use standard OpenShift security constraints and include a
`p4-gpu:NoSchedule` toleration. Review them for your cluster's scheduling policy.

## Install the tooling

From the repository checkout:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
mistral-validate --help
```

No Helm or local GPU libraries are required. The commands below use Make from
the repository root with the virtual environment active. The installed
`mistral-validate` CLI also works outside the checkout;
`python -m mistral_validation` is the equivalent module entry point.

## Configure the deployment

Choose a dedicated namespace and provision its pull secret through your normal
secret-management process before running preflight. Never put entitlement keys
or tokens in the repository or Make variables. Preflight selects an existing
namespace or creates it when absent, but does not create the pull secret.

| Input | Default | Purpose |
| --- | --- | --- |
| `NAMESPACE` | `mistral-custom-model-ocp` | Dedicated deployment namespace |
| `PULL_SECRET` | `ibm-entitlement` | Existing namespace-scoped IBM pull secret |
| `GPU_COUNT` | `8` | GPUs for one serving pod and tensor parallel size |
| `PVC_NAME` | `mistral-weights` | Existing or new model PVC |
| `PVC_SIZE` | `140Gi` | Requested capacity; at least 140Gi |
| `STORAGE_CLASS` | cluster default | Storage class for the PVC |
| `LAYOUT` | `.local/2411-layout.json` | Private model inventory |
| `PREPARE_TIMEOUT` | `14400s` | Inventory and preparation timeout |
| `SERVING_TIMEOUT` | `3600s` | Startup allowance and readiness wait |
| `REPORT_DIR` | `reports` | Local operation reports |
| `CLEANUP_TIMEOUT` | `300s` | Timeout per cleanup deletion or API request |
| `OC` | `oc` | OpenShift CLI binary |
| `PYTHON` | `python3` | Python binary |

Make accepts exported variables as well as command-line overrides; command-line
values take precedence. Keep the same namespace, pull secret, PVC, and GPU
configuration throughout the deployment. For example:

```sh
make preflight NAMESPACE=mistral-custom-model-ocp PULL_SECRET=ibm-entitlement \
  GPU_COUNT=8 PVC_NAME=mistral-weights PVC_SIZE=140Gi
make render-config NAMESPACE=mistral-custom-model-ocp PULL_SECRET=ibm-entitlement \
  GPU_COUNT=8 PVC_NAME=mistral-weights PVC_SIZE=140Gi
```

`render-config` prints the effective configuration and the five pinned image
references without contacting a cluster. Pass the same overrides used for
preflight when previewing a customized configuration.

## Prepare the model

Inventory scans the four pinned weight images and writes a private file list
with sizes and SHA-256 values. Review it, then prepare the model PVC:

```sh
make inventory NAMESPACE=mistral-custom-model-ocp
${EDITOR:-vi} .local/2411-layout.json
make prepare NAMESPACE=mistral-custom-model-ocp
```

Preparation retains the model's separate safetensors shards and publishes the
completed model directory only after every inventoried file passes size and hash
verification. Identical duplicate files are copied once; conflicts fail.

Preparation creates the PVC when absent. Retry a failed or interrupted attempt
with the same command after resolving the cause; an existing completed model
directory is preserved. Run preparation while serving is stopped. If capacity
is insufficient, expand the claim or select a larger `PVC_SIZE` before retrying.

Keep `.local/`, rendered preparation output, and `reports/` private and out of
Git. Preparation previews contain the full inventory and file paths. For offline
previews, inventory details, and retry behavior, see the
[configuration reference](docs/reference.md#model-preparation-and-integrity).

## Deploy and check inference

Preview the serving resources, then install with the same configuration:

```sh
make -s render-deployment NAMESPACE=mistral-custom-model-ocp > /tmp/mistral-serving.json
make install NAMESPACE=mistral-custom-model-ocp
make status NAMESPACE=mistral-custom-model-ocp
```

Install checks that preparation completed on the selected PVC, including its
immutable UID, then applies the ServingRuntime and InferenceService. It waits
for readiness and checks chat inference through the KServe Service. The prepared
model is mounted read-only. Retain the completed preparation Job, its pod, and
verification logs; installation checks this evidence against the current claim.

Startup time depends on storage throughput and cache state. Increase
`SERVING_TIMEOUT` if necessary. See the
[runtime reference](docs/reference.md#serving-behavior-and-runtime-compatibility)
for the loader settings, readiness checks, and preparation identity requirements.

For direct CLI use, the equivalent install command is:

```sh
mistral-validate install --namespace mistral-custom-model-ocp \
  --pull-secret ibm-entitlement --pvc-name mistral-weights \
  --gpu-count 8 --serving-timeout 3600s
```

The runtime serves `mistralai/mistral-large-instruct-2411` with a configured
maximum context length of 32768 and maximum sequence count of 20. Use the
[extended checks](docs/verification.md) to assess concurrent requests, near-limit
context inference, and restart recovery on your deployment.

## Check the protected external route

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
The standalone `make validate` report under `reports/` records status codes,
the outcome, the exact prompt once a chat request is attempted, and the
assistant text if the authenticated response passes validation. Unavailable
prompt or response values are `null`. Set `REPORT_DIR` to change the report
location. TLS certificate verification remains enabled. A failed check returns
a nonzero status and identifies whether the route, authentication, readiness,
or inference needs attention.

## Uninstall and remove model data

To stop serving and remove the installation workloads, run:

```sh
make uninstall NAMESPACE=mistral-custom-model-ocp PVC_NAME=mistral-weights
```

`uninstall` removes the InferenceService and its dependent serving resources,
the ServingRuntime, preparation and inventory discovery Jobs, and preparation
ConfigMap. It waits for foreground deletion so dependent pods stop before it
reports success. The namespace, pull secret, PVC, and prepared model files are
preserved. It also removes the named predictor Deployment, Service, route, and
auth/dashboard ConfigMaps if left over from an earlier deletion.

To also delete the model claim and its weights, explicitly run:

```sh
make purge NAMESPACE=mistral-custom-model-ocp PVC_NAME=mistral-weights
```

`purge` prints the target namespace and claim before deletion. It requires the
installation workloads to have been removed by `uninstall` and refuses to delete
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
Job used as installation evidence; to reinstall, purge and repeat inventory,
preparation, and installation.

## Troubleshooting

| Symptom | Operator action |
| --- | --- |
| Preflight fails | Resolve its permissions, secret, quota, storage or GPU placement failure; GPUs already allocated to another service are unavailable. |
| Image pull fails | Inspect `oc describe pod -n <namespace> <pod>` locally; verify entitlement covers all five pinned images and the secret is in this namespace. Never print secret data. |
| Preparation fails | Stop serving, inspect Job events and fixed failure markers, confirm capacity and inventory, then rerun `make prepare` with identical inputs. A completed directory is deliberately preserved. |
| Loading or readiness fails | Use `make status`, inspect predictor events and logs locally, and the [runtime reference](docs/reference.md#serving-behavior-and-runtime-compatibility). Allow time for a cold load; use `make runtime-check` for the loader diagnostic. |
| Route authentication fails | Check external HTTPS URL, platform authentication and token permissions/expiry, then rerun `make validate`. Keep TLS verification enabled. |
| Recovery or capacity check fails | Inspect its JSON result, predictor readiness and local logs; rerun the individual command after resolving the cause. Use the [extended verification guide](docs/verification.md) for report details. |
| Cleanup fails | Resolve resource access/finalizer or remaining-pod issues and repeat explicit cleanup. Never force finalizers away. |

## Further documentation

- [Configuration, preparation, and runtime reference](docs/reference.md)
- [Extended deployment verification](docs/verification.md)
- [Development and local checks](docs/development.md)
