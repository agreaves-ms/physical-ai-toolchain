---
description: 'Essential development conventions for microsoft/physical-ai-toolchain.'
applyTo: '**'
---

# Repository Instructions

Follow the affected area's existing conventions and scoped instructions. Use manifests, lockfiles, and tool configuration for current versions, commands, and lint rules rather than duplicating them here.

## Editing

* Preserve user changes and explicit scope limits. Do not add backward-compatibility layers unless requested.
* Prefer coherent root-cause fixes across the affected solution. Refactor when SOLID, KISS, or clearer responsibilities improve the design; use "1, 2, refactor" as a checkpoint, not a requirement to invent abstractions.
* You may fix concrete issues discovered outside the original request. Validate those fixes and report them separately; avoid speculative cleanup.
* Keep comments brief and factual. Avoid narration and plan-phase markers.
* Do not modify vendored files in `external/`.

## Python

* Use `uv`, not pip. Preserve the affected project's Python constraints.
* Put `from __future__ import annotations` first among imports. Fully annotate function parameters and returns; do not annotate local variables.
* Regenerate `uv.lock` after dependency changes. Do not hand-edit locks or commit derived flat requirements files.

## Validation

* Use focused tests and probes during development to refine logic and catch defects.
* After completing a coherent behavior, run the relevant checks and supported automatic fixes before manually addressing remaining lint issues. Avoid repeated per-file style cleanup while the behavior is still taking shape.
* Respect configured exclusions; do not bypass them with alternate configurations or forced file selection. Keep review-only checks non-mutating and report unavailable or failed checks.

## Documentation

Write natural, direct, reader-focused prose. Avoid corporate filler, inflated claims, and repetitive structure. Improve problematic documentation as a whole, moving information to its appropriate home and updating links rather than only polishing sentences.

## Agent Workflows

* The primary agent writes and updates RPI plans; do not use `RPI Planner` subagents.
* Delegate RPI reviews and critiques to `GPT-5.6 Luna (copilot)`, unless the user explicitly chooses another model for `rpi-review` or `rpi-plan-critique`. The caller may provide context and accept, disregard, or verify findings; it owns the final output.
* HVE Builder does not require subagent reviews. Do not use `HVE Artifact Tester`; the primary agent performs surface-level `hve-builder-tester` checks and states what remains unverified. Separate RPI reviews still follow the rule above.

## Companion Library

`microsoft/physical-ai-toolchain-skills` enables agentic scenarios for this toolchain. When working across repositories, locate its checkout and follow its `AGENTS.md` and relevant capability contracts. Do not assume a fixed skill inventory or copy implementations between repositories.
