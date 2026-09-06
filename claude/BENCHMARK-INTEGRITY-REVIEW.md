# Benchmark acceptance and timing correction

Baseline `31e1bcf86285a0749ca3dd8167c40efadf14cdcb` accepted a wrong count,
an incomplete variant set, invalid timing and a synthetic GPU error with exit 0.
Its `endToEndMs` added upload plus compute, excluding result readback. Variant E
also omitted its initial resident upload. The final comparison selected the best
compute variant and reused it as the supposedly best end-to-end variant.

The corrected entrypoint rejects those results before emitting benchmark JSON,
requires 2–100 iterations, and records the input and executing source hashes.
It measures each iteration through result readback, adds resident setup once,
and reports the full measured operation total divided by all iteration counts.
Warm upload/compute columns still discard iteration 1 and remain separate from
the full-operation total. Startup, file I/O, pipeline compilation and CPU checking
are explicitly outside that GPU operation timing. Both WGSL strings are unchanged.

Six synthetic CLI regressions verify rejection and accounting. The distinguishing
fixture has A as fastest compute and B as fastest full operation; E's setup 10 ms
plus five 2.5 ms iterations must report 22.5 ms total / 4.5 ms per operation.
These synthetic fixtures are not hardware measurements.

A separate bounded actual run used 1,048,589 bytes and five iterations per variant.
All 25 GPU counts were 4,099, matching two CPU methods. Installed Chrome headless
with no extra flags selected Intel `xe-lpg`. The evidence therefore makes no new
RTX 5070 performance claim. A single sequential run also does not establish a
general speedup, noise distribution or optimal variant.

Raw source-attributed stdout, readbacks and hashes are under
`offline-screenshots/recovery-20260906/gpu-benchmark-{negative,positive,hardware}`.
Historical result files are unchanged. The existing `codex/overnight.py` consumer
uses the preserved `rows[].correct` field; it does not recompute the timing field.
This branch is a draft review for the owner, separate from the corpus-readback fix.
