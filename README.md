# nvidia-drivers-for-global-grid

Does the NVIDIA hardware actually earn its place in Global Grid's compute path?
This repository answers that with measurements, not vendor claims.

## Lanes

Work here is organised into independent lanes. A lane is self-contained: it owns
its own harness, its own results and its own conclusions, and does not read from
or depend on any other lane. Two lanes reaching the same answer by different
routes is evidence; two lanes sharing code is not.

| Lane | What it holds |
| --- | --- |
| [`claude/`](claude/) | GPU-vs-CPU benchmark harness — a CPU/RAM worker-thread ladder and a WebGPU compute bench that turns on one NVIDIA CUDA C++ Best Practices rule at a time, with measured results from an RTX 5070 Laptop GPU on 2026-09-05 |

### The `claude/` lane in one line

On a 26.29 MB text artefact the RTX 5070 **loses end-to-end** to a single line of
`Buffer.indexOf` (x0.38) — the PCIe upload costs more than the whole CPU parse.
Only NVIDIA's High-Priority "minimise host↔device transfer" rule paid off: keep
the buffer resident and run many kernels, and it is x3.15. Full tables, the
variants that turned out to be noise, and the two Chromium/WebGPU gotchas are in
[`claude/README.md`](claude/README.md); the raw measured JSON is in
[`claude/results/`](claude/results/).
