# Measured results — 2026-09-05

Committed evidence, not illustration. Every number in these files is a timing
taken on one named machine on one named artefact.

| File | What it holds |
| --- | --- |
| `2026-09-05-machine.json` | The box, the two GPUs, the driver, and the artefact these runs ate |
| `2026-09-05-cpu-ram-ladder.json` | Single-thread phase timings and the 1/2/4/8/12-thread ladder |
| `2026-09-05-gpu-rtx5070.json` | The five rule variants, the adapter that answered, and the CPU comparison |
| `corpus-gpu-analysis-202609051700.json` | The GPU corpus analysis: two-pass timings, the SHA-256 duplicate groups, and the top 100 near-duplicate pairs from the similarity screen |

**Machine:** Intel Core Ultra 7 255HX (20 cores / 20 logical), 15.46 GB
DDR5-6400 (2 x 8 GB), GeForce RTX 5070 Laptop GPU (8151 MiB, driver
592.02, compute capability 12.0) alongside an Intel iGPU. Node v24.19.0,
Windows 11.

**Artefact:** a 27,568,130-byte (26.29 MB) GridAtlas teleprint source-code dump —
112 BEGIN/END section markers, 43,820 `=` bytes. It is **not committed**: it is
evidence, and 26 MB does not belong in git. `../make-sample-artefact.mjs` writes
a same-shaped stand-in so the harness runs anywhere; numbers from the stand-in
are not these numbers.

**Corpus (`corpus-gpu-analysis-202609051700.json` only):** a different input
from the teleprint — the GridAtlas served tree `atlas/`, 223 files,
40,161,723 bytes (38.30 MB), `.js`/`.mjs`/`.html`/`.css` only. The corpus is
**not committed** either; it is a directory of someone else's repository and
this lane only reads it.

The file is 26 KB and is committed whole — the `nearDuplicates.top` array holds
the first 100 pairs the analyser itself caps at, and nothing has been trimmed.

## Reading `corpus-gpu-analysis-202609051700.json` without misreading it

The file reports **two independent numbers that must never be conflated or
added together**:

- `exactDuplicates` — 12 groups, 487,454 redundant bytes (0.46 MB), 1.2% of the
  corpus. This is a **fact**, established on the host by SHA-256. These files
  are byte-identical.
- `nearDuplicates` — 11,053 of 24,753 pairs at cosine ≥ 0.99, of which 77 are
  byte-identical and 10,976 are near-but-not-identical. This is a **screen**,
  not a proof.

**Cosine similarity on 256-bin byte histograms is a screen, not a proof of
duplication.** A low score proves difference; a high score only says "look
here". Two 363 KB files scoring 1.00000 share a *byte distribution*, which is
not the same as sharing bytes. Nothing in `nearDuplicates` establishes that any
byte is redundant.

There is deliberately **no "bytes saveable" figure in this file.** The obvious
sum — bytes sitting in near-duplicate pairs, ~1976 MB on this corpus — is a
meaningless upper bound, because pairs overlap heavily and a single file is
counted once per pair it appears in; the total exceeds the whole 38 MB corpus
by fifty times. The analyser prints it to the console with that warning
attached and keeps it out of the JSON so it cannot be picked up as data. The
only defensible redundancy number here is the SHA-256 one: **0.46 MB**.

These results are from the `claude/` lane only. They say nothing about, and
assume nothing about, any other lane in this repository.
