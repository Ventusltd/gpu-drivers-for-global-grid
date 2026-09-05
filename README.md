# gpu-drivers-for-global-grid

Does a discrete GPU actually earn its place in Global Grid's compute path?
This repository answers that with measurements, not vendor claims.

This repository is independent. It is not affiliated with, endorsed by, or
associated with any GPU vendor. Hardware and vendor names appear below only
where they identify what was measured — the adapter string a driver returned,
the product name of the card in the machine — and nowhere else.

## Lanes

Work here is organised into independent lanes. A lane is self-contained: it owns
its own harness, its own results and its own conclusions, and does not read from
or depend on any other lane. Two lanes reaching the same answer by different
routes is evidence; two lanes sharing code is not.

| Lane | What it holds |
| --- | --- |
| [`claude/`](claude/) | GPU-vs-CPU benchmark harness — a CPU/RAM worker-thread ladder, a WebGPU compute bench that turns on one CUDA C++ Best Practices Guide rule at a time, and a two-pass GPU corpus analysis, with measured results from an RTX 5070 Laptop GPU on 2026-09-05 |

### The `claude/` lane in one line

On a 26.29 MB text artefact the RTX 5070 **loses end-to-end** to a single line of
`Buffer.indexOf` (x0.38) — the PCIe upload costs more than the whole CPU parse.
Only the guide's High-Priority "minimise host↔device transfer" rule paid off: keep
the buffer resident and run many kernels, and it is x3.15. Full tables, the
variants that turned out to be noise, and the two Chromium/WebGPU gotchas are in
[`claude/README.md`](claude/README.md); the raw measured JSON is in
[`claude/results/`](claude/results/).

### And where it wins

The same lane's corpus analysis is the counter-case: 38.30 MB of served
GridAtlas code crosses PCIe once in 37.6 ms, then per-file byte histograms take
4.10 ms and **24,753 pairwise cosine similarities take 40 ms**. Quadratic work
over resident data is where the card earns its place. The result is a *screen*,
not a proof — cosine similarity on byte histograms says "look here", never "these
are duplicates"; exact duplication is a separate SHA-256 fact (0.46 MB, 1.2% of
the corpus) and the two numbers are never added together.
