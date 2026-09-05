# Measured results — 2026-09-05

Committed evidence, not illustration. Every number in these files is a timing
taken on one named machine on one named artefact.

| File | What it holds |
| --- | --- |
| `2026-09-05-machine.json` | The box, the two GPUs, the driver, and the artefact these runs ate |
| `2026-09-05-cpu-ram-ladder.json` | Single-thread phase timings and the 1/2/4/8/12-thread ladder |
| `2026-09-05-gpu-rtx5070.json` | The five NVIDIA-rule variants, the adapter that answered, and the CPU comparison |

**Machine:** Intel Core Ultra 7 255HX (20 cores / 20 logical), 15.46 GB
DDR5-6400 (2 x 8 GB), NVIDIA GeForce RTX 5070 Laptop GPU (8151 MiB, driver
592.02, compute capability 12.0) alongside an Intel iGPU. Node v24.19.0,
Windows 11.

**Artefact:** a 27,568,130-byte (26.29 MB) GridAtlas teleprint source-code dump —
112 BEGIN/END section markers, 43,820 `=` bytes. It is **not committed**: it is
evidence, and 26 MB does not belong in git. `../make-sample-artefact.mjs` writes
a same-shaped stand-in so the harness runs anywhere; numbers from the stand-in
are not these numbers.

These results are from the `claude/` lane only. They say nothing about, and
assume nothing about, any other lane in this repository.
