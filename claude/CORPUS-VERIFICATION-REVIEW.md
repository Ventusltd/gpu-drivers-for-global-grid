# Corpus result acceptance review

Baseline owner `31e1bcf86285a0749ca3dd8167c40efadf14cdcb` ran the same analyser CLI
with an explicitly synthetic browser readback. A wrong histogram printed
`DISAGREES` but returned exit 0. Incorrect, null and truncated similarity
readbacks also returned exit 0. These fixtures test result acceptance, not GPU
execution, and say nothing about whether the historical measured results were wrong.

The change checks both complete arrays against the CPU, rejects WebGPU validation
errors and invalid timing/count fields, and rejects an all-empty corpus before
launching a browser. Invalid results return exit 1 and write a rejected report
without timings or similarity candidates, replacing a possibly stale output.
Successful reports retain the adapter and SHA-256 of the executing harness,
concatenated corpus, and every input file. Both WGSL shaders are unchanged.
The cosine tolerance is an absolute 0.00002; classification within that distance
of the 0.99 screen threshold remains uncertain. CPU verification is outside the
timed GPU stages. No performance improvement is claimed.

Ten CLI tests cover good and corrupted readbacks, WebGPU errors, timing/count
validation, and all-empty input. A separate actual hardware check used seven
files / 66,309 bytes: empty, one-byte, unaligned, all 256 values, an exact duplicate,
a reversed distribution, and a skewed 65,537-byte file. Installed Chrome headless
with no extra browser flags selected `vendor=intel`, `architecture=xe-lpg`.
All 1,792 histogram bins and 49 similarity entries passed; this is Intel evidence,
not a new RTX 5070 measurement. Its launch-only Playwright adapter changed no
application, reference or shader code.

Raw evidence is under `offline-screenshots/recovery-20260906/`:
`gpu-verification-negative`, `gpu-verification-positive`, and
`gpu-verification-hardware-final`. Historical `claude/results/` files are untouched.
The existing corpus-review launch adapter remains compatible because the analyser
still has one standalone entrypoint and the original launch expression. No owner
main merge is authorized by this draft; review and CI acceptance remain separate.
