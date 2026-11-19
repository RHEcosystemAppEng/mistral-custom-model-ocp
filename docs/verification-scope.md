# Approved 2411 verification scope

The source specification is
[mistral-validation issue #1](https://github.com/TomerG2/mistral-validation/issues/1).
This document records the operator confirmations supplied in local ticket 06,
"Record the approved 2411 verification scope and human review": retain separate
2411 shards according to IBM's clarification, permit merging with relevance
review pending, and record the later human-review outcome as "Mostly relevant."
These confirmations amend the original specification locally; this document
does not modify or close the parent issue.

## Approved preparation workflow

For `mistral-large-instruct-2411`, the operator approved IBM's clarification that
the model keeps separate safetensors shards. The source run's discovery observed
26 `model-*-of-00026.safetensors` shards across the four pinned weight images.
Preparation uses overlay assembly, verifies file sizes and SHA-256 values against
the private inventory, and publishes the completed model directory only after
verification. Serving mounts that directory read-only.

This supersedes the original 2411 requirement to concatenate parts into
`consolidated.safetensors`, including user story 10 and the corresponding
preparation implementation decision in the source specification. The verified,
complete model directory remains the preparation gate. The clarification does
not establish a workflow for other model versions or replace extraction checks
with file existence alone. See the [preparation runbook](../README.md#prepare-the-2411-artifacts).

## Human-review addendum

The operator approved merging while relevance review was pending and subsequently
stated that a human review had occurred, describing the answers as **"Mostly
relevant"**. This is a qualified qualitative outcome, not an unconditional
answer-quality pass.

| Review metadata | Known value |
| --- | --- |
| Reported outcome | Mostly relevant |
| Sample count | Unknown |
| Review UTC timestamp | Unknown |
| Reviewer identity | Unknown |
| Association with a recorded run | Unknown; not assigned to the ai-dev04 run |
| Run-specific review evidence | Pending |

This later statement is an addendum, not a result recorded during the
2026-11-18 run. The [historical summary](evidence/ai-dev04-20261118.md) and
[machine-readable bundle](evidence/ai-dev04-20261118.json) remain unchanged:
they record human review as pending, including a recorded sample count of zero.
That zero describes the historical report, not the unknown sample count of the
later review. Do not backfill a reviewer, timestamp, sample count, or run identity
from this statement, or change machine-readable verification to passed on its
strength. Merge permission does not waive pending evidence or change CLI results.

For future run-specific reviews, follow the [evidence runbook](../README.md#record-one-run-and-its-limits)
and record only approved review metadata and the conclusion. No prompts, answers,
model contents, credentials, or private inventories are included in this addendum.

## Limits of the recorded verification

The automated evidence comes from the source repository before this checkout's
bootstrap. It covers the recorded ai-dev04 configuration: OCP 4.22.20, RHOAI 3.5.2,
eight NVIDIA A100-SXM4-40GB GPUs, the pinned image set and configuration hash in
the linked bundle, tensor parallel size 8, model length 32768, max sequences 20,
and a 140Gi gp3-csi PVC.

The recorded behaviors include preparation interruption/retry, readiness and
authenticated route inference, restart recovery, 20 concurrent requests, a
separate near-32K context request, uninstall, and explicit PVC/backing-PV removal.
The capacity checks were separate; they establish neither combined stress
capacity nor long-context recall or general answer quality. The later qualitative
review does not broaden these measurements to other configurations. This
documentation work performs no new live validation of this checkout.

## Proposed correction for the source issue

The following is prepared for follow-up on
[the parent specification](https://github.com/TomerG2/mistral-validation/issues/1);
it has not been posted or applied to that issue by this work.

> Replace user story 10's consolidated-model requirement with: As an operator, I
> want preparation to assemble and verify the complete 2411 model directory
> while retaining its separate safetensors shards, so that serving never reads
> incomplete artifacts.
>
> In the preparation implementation decision, replace reconstruction of
> `consolidated.safetensors` with verified overlay assembly of the four pinned
> weight images. Publish the complete directory only after inventory-based file
> size and SHA-256 checks succeed; retain safe retry and read-only serving.
> Update related concatenation/reconstruction wording and the open checksum
> question to reflect the approved 2411 separate-shard layout and its integrity
> checks, rather than requiring a checksum for a nonexistent consolidated file.
>
> Record the approved merge exception for pending relevance review separately
> from verification status. Record the later operator conclusion, "Mostly
> relevant," as a qualified addendum with sample count, review timestamp,
> reviewer identity and run association unknown. Preserve historical pending
> results and do not infer an unconditional machine-readable verification pass.
