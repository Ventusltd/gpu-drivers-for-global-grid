# Restart lesson: reuse the established GPU method

25 September 2026. Recorded at the owner's request after an avoidable agent error.

## What went wrong

The assistant proposed designing a GPU/CI runner arrangement before inspecting the existing handoff, runner estate and reusable computation code. The owner had already established the electron/positron method, documented it and repeatedly asked for reuse. The assistant's framing implied missing infrastructure and repeated work instead of identifying the actual integration gap.

## Required restart sequence

1. Read docs/GPU-TESTING-LESSONS-20260924.md and the existing engine/fused.py implementation before proposing a new harness. Consult additional private circuit-model lessons where available; do not publish them as part of this public note.
2. Inspect the worlds- source and receipts: src/annihilate.py, night-results/annihilate.json and the corrected aggregate receipt referenced in the testing lessons. Read the full current handoff, including later updates; its opening paragraph can predate appended evidence.
3. Inspect existing runner registrations, labels, live online/offline status and workflows before claiming runner infrastructure is missing. Existing registrations were found; the twelve domain runners were offline when checked. Offline is not absent. Local GPU execution still worked.
4. Preserve paused workflows and unfinished work. The existing weekly-card workflow and weekly_card.py in the domain working copy were uncommitted adjacent work when inspected; do not silently publish or replace them.
5. Adapt an existing measured GPU pattern to the actual workload, run bounded checks against an independent reference, and retain compact receipts. Use agent reasoning for algorithms, code review and failures, not repetitive numeric enumeration.

## Established pattern to retain

Keep arrays resident, generate large index spaces on-device, fuse appropriate arithmetic, reduce before transfer, and retain bounded discrepancies/counters rather than per-case dumps. Reuse the corrected kernel namespace handling. Separate unique cases, predicate evaluations and repeated soak passes. Synchronize timing and report preparation/transfer versus warm GPU intervals separately. Historical electron/positron strict-versus-inclusive predicates detect boundaries; they are not universally independent algorithms.

## Current concrete evidence

The bounded Pipeline News probe used 7,680 actual records and 1,024 filter combinations, 7,864,320 record-query pairs per pass. All masks/counts/aggregate comparisons passed, with 16 original-record scalar witnesses and 16 top-20 comparisons plus nonfinite/boundary fixtures. Eleven alternating samples measured 38.3314 ms NumPy CPU median and 0.4823 ms synchronized GPU wall median for warm resident numeric filtering/counts/capacity sums. Load/encoding was 0.0643 seconds; GPU preparation/warmup 0.1406 seconds. Ranking, setup, transfers and browser/network were excluded from the warm comparison. This is a bounded numerical result, not proof of billion-record database throughput or a whole-CI speedup.

## Actual remaining integration task

Reuse the existing runner/receipt mechanism to connect a declared workload and verified input/output hashes to CI. Do not build a parallel platform merely because the existing one was not inspected. Do not assume historical runner registration means it is running now. Private source locations and confidential inputs stay outside public notes and artifacts.
