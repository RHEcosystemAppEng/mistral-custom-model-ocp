# Extended deployment verification

Start with the [installation guide](../README.md). These checks are optional for
a basic deployment. Restart verification deletes the serving pod; the complete
workflow also interrupts preparation and requires explicit cleanup. Run them
when disruption is acceptable.

Use the same namespace, PVC, image set, and configuration throughout a run.
For external-route checks, supply `RHOAI_TOKEN` as described in the installation
guide. Keep it exported until the checks finish, then run `unset RHOAI_TOKEN`.
Reports are written under `reports/` by default; set `REPORT_DIR` to override it.
Keep reports private and out of Git.

## Restart recovery

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

## Concurrent requests

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
and model logs as appropriate. Run this check against the effective deployment to assess its capacity.

## Near-32K context

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
and model logs as appropriate. Run this check against the effective deployment to assess its near-32K capacity.

## Complete verification workflow

Use one namespace, claim, image set and configuration throughout a run. The
individual installation and verification commands can be run separately; this wrapper executes them
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
then follow the [troubleshooting guide](../README.md#troubleshooting).
Retain the complete private bundle.

Automated HTTP and structural response checks do not establish answer relevance.
The wrapper leaves human review pending and exits nonzero while review or cleanup
is outstanding. There is no CLI command to complete the human-review stage.
For a separate human assessment, inspect responses to short synthetic questions
with known answers and record only the review timestamp, sample count, reviewer
identifier, and a conclusion of relevant, mixed, or irrelevant. Keep prompts and
responses out of the repository. A separate assessment does not change the
wrapper's machine-readable pending result.

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
