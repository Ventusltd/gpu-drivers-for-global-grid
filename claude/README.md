# `claude/` lane — GPU vs CPU, measured

A working benchmark harness that answers one question with timings rather than
belief: **for the work Global Grid actually does to a large text artefact, is
the discrete NVIDIA GPU worth using?**

Measured on one named machine on 2026-09-05. Every figure below is a timing this
harness took; nothing here is a vendor claim or an extrapolation.

This lane is self-contained. It does not touch, read, or assume any other lane
in this repository.

---

## The machine and the artefact

| | |
| --- | --- |
| CPU | Intel Core Ultra 7 255HX — 20 cores / 20 logical |
| RAM | 15.46 GB DDR5-6400 (2 x 8 GB) |
| GPU (discrete) | NVIDIA GeForce RTX 5070 Laptop GPU — 8151 MiB, driver 592.02, compute capability 12.0 |
| GPU (integrated) | Intel iGPU — present, which is exactly why the answering adapter is printed |
| OS / runtime | Windows 11, Node v24.19.0 |
| Artefact | GridAtlas teleprint source-code dump — 27,568,130 bytes (26.29 MB), 112 BEGIN/END markers, 43,820 `=` bytes |

The artefact itself is **not committed**. It is evidence, not source, and 26 MB
does not belong in git. `make-sample-artefact.mjs` writes a same-shaped
stand-in so the harness runs on any machine, including CI.

---

## What the lane measures

### 1. `bench-cpu-ram.mjs` — the CPU/RAM ladder

Four phases, kept separate so a slow one is visible rather than averaged away:
file **read** (storage + page cache), **memcpy** (DDR5 bandwidth, one core),
**sha256** (compute-bound), **utf8 decode** and **regex parse** of every
`BEGIN`/`END` section (branch- and string-bound).

Then the same work on 1..N worker threads where **each worker reads and holds
its own copy**. That is the honest model of N readers opening the artefact on a
DC machine — not one shared buffer sliced N ways, which would measure something
else entirely.

### 2. `bench-gpu.mjs` — five NVIDIA rules, one at a time

The same artefact uploaded to the discrete GPU and reduced by a WebGPU compute
shader, driven through Playwright Chromium. The work is counting bytes equal to
`=` (0x3D) — not arbitrary: the teleprint's section boundaries are runs of `=`,
so this is the first pass of a real parse, expressed as the embarrassingly
parallel reduction a GPU exists for.

Five variants, each turning on **one** rule from NVIDIA's CUDA C++ Best
Practices Guide, so the cost of each is a measured delta rather than a belief:

| | Variant | NVIDIA rule under test |
| --- | --- | --- |
| A | `writeBuffer` upload, one u32 per thread, workgroup 256 | baseline — nothing beyond one batched transfer |
| B | upload via `mappedAtCreation` | page-locked/pinned memory attains the highest transfer bandwidth |
| C | one `vec4<u32>` (128 bits) per thread | coalesced 128-bit loads maximise global memory throughput |
| D | fewer, fatter workgroups with a grid-stride loop | occupancy and ILP hide memory latency |
| E | upload **once**, run the kernel N times | **High Priority:** minimise host↔device data transfer |

**Every variant is checked against the CPU ground truth.** A fast wrong answer
is not a result. All five agreed on 43,820.

---

## Measured: the CPU/RAM ladder

Single thread, on the 26.29 MB artefact:

| Phase | Time | Throughput |
| --- | --- | --- |
| read | 8.2 ms | ~3.2 GB/s |
| memcpy | 7.6 ms | ~3.4 GB/s |
| sha256 | 6.4 ms | ~4.1 GB/s |
| utf8 decode | 14.7 ms | — |
| regex parse | 7.5 ms | — |

N independent readers, each with its own resident copy:

| Threads | Wall | Aggregate | Peak worker RSS | Speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 73 ms | 360 MB/s | 246 MB | x1.00 | 100% |
| 2 | 77 ms | 685 MB/s | 367 MB | x1.90 | 95% |
| 4 | 89 ms | 1185 MB/s | 607 MB | x3.29 | 82% |
| 8 | 120 ms | 1750 MB/s | 1080 MB | x4.86 | 61% |
| 12 | 168 ms | 1874 MB/s | 1516 MB | x5.21 | 43% |

**Scaling rolls off hard after 8 threads.** Free RAM at the start was 4.42 GB
and 12 workers held 1.5 GB resident, so the ceiling here is memory and memory
bandwidth — not cores. Twenty logical cores were available; the machine ran out
of bandwidth and headroom long before it ran out of them.

---

## Measured: the RTX 5070

Adapter that actually answered: `vendor=nvidia architecture=blackwell
device=0x2d58` — printed, not assumed.

CPU reference on the same 26.29 MB:

| | Time | Throughput |
| --- | ---: | ---: |
| CPU scalar byte loop | 15.3 ms | 1719 MB/s |
| CPU `Buffer.indexOf` | 3.6 ms | 7227 MB/s |

The five variants (warm means, first iteration discarded):

| | Variant | Upload | Compute | Correct? |
| --- | --- | ---: | ---: | :---: |
| A | baseline | 7.18 ms (3664 MB/s) | 2.350 ms (11188 MB/s) | yes |
| B | mapped / pinned | 9.93 ms (2649 MB/s) | 3.175 ms (8281 MB/s) | yes |
| C | vec4 coalesced | 11.03 ms (2385 MB/s) | 3.050 ms (8620 MB/s) | yes |
| D | grid-stride | 11.93 ms (2205 MB/s) | 2.725 ms (9648 MB/s) | yes |
| E | resident | 0 (uploaded once) | 3.025 ms (8691 MB/s) | yes — **x3.15 end-to-end** |

GPU against CPU:

| Comparison | Ratio |
| --- | ---: |
| compute-only, GPU vs CPU scalar loop | x6.5 |
| compute-only, GPU vs CPU `Buffer.indexOf` | x1.5 |
| **end-to-end, GPU vs CPU `Buffer.indexOf`** | **x0.38** |

---

## The honest conclusion

**For a single pass over a ~26 MB artefact, the RTX 5070 is not worth it.**
The PCIe upload costs 7–12 ms; the entire CPU parse costs 3.6 ms. End-to-end
the GPU comes in at **x0.38** — it loses, and it loses to one line of
`Buffer.indexOf`.

Of the five NVIDIA rules, **only the High-Priority one paid**: minimise
host↔device transfer. Keeping the buffer resident and running many kernels over
it (variant E) was x3.15 end-to-end against the baseline.

**B, C and D did not beat the baseline, and they must not be reported as wins.**
At 2.3–3.2 ms the kernel is already at the submission-overhead floor and is
memory-bound, so pinned uploads, 128-bit coalesced loads and grid-stride
occupancy have nothing left to buy. Those deltas are within noise. Applying a
rule from a good guide is not the same as measuring a gain from it.

**The design implication for a DC machine** is therefore narrow and specific:
the GPU becomes the right tool only when the data **stays resident** and **many
passes run over it**. A pipeline that uploads an artefact, does one reduction
and drops it will be slower than the CPU, however good the kernel is.

### Two environment gotchas, both measured here

1. **Headless Chromium returns no WebGPU adapter on this machine.**
   `requestAdapter()` returns `null` with *"No available adapters"*. Run the GPU
   bench headless and it will confidently report "no GPU" on a laptop that has
   two. The bench therefore runs **headed** by default; `--headed=0` exists only
   so you can reproduce the failure.
2. **Chrome ignores `powerPreference` on Windows**
   ([crbug.com/369219127](https://crbug.com/369219127)). Asking for
   `high-performance` guarantees nothing. The adapter that actually answered
   must be **printed and checked** — otherwise you may be benchmarking the Intel
   iGPU and calling it an RTX 5070.

---

## How to run it

```sh
# 0. optional: make a stand-in artefact if you do not have the teleprint
node claude/make-sample-artefact.mjs sample-artefact.txt 26

# 1. the CPU/RAM ladder
node claude/bench-cpu-ram.mjs <artefact.txt> --threads 1,2,4,8,12

# 2. the GPU bench (headed; needs Playwright's Chromium and a WebGPU adapter)
node claude/bench-gpu.mjs <artefact.txt> --iters 5
```

The artefact path is optional in both. Resolution order: the first argument,
then `$BENCH_ARTEFACT`, then `sample-artefact.txt` next to the script.

`bench-gpu.mjs` needs Playwright. Set `PLAYWRIGHT_PATH` to an existing
`playwright/index.js`, or `npm i -D playwright && npx playwright install
chromium` and it will resolve the bare specifier. `BENCH_CHANNEL=chrome` uses
installed Chrome instead of Playwright's Chromium.

### CI

`.github/workflows/claude-bench.yml` runs the CPU/RAM ladder on a
GitHub-hosted runner against a generated stand-in artefact, on push to this
lane and on `workflow_dispatch`. It **skips the GPU bench with an explicit
message**: GitHub-hosted runners have no NVIDIA GPU, and a green tick on a
benchmark that never ran would be worse than no tick at all. The GPU numbers in
`results/` can only come from a machine with the card.

One caveat when reading any ladder, CI's included: the wall clock for each rung
includes worker **start-up**, which is a fixed cost the 1-thread rung pays in
full and cannot amortise. On a fast runner that can push the 2-thread rung above
100% efficiency — an artefact of the baseline, not super-linear scaling. It does
not affect the measured table above, where 1 thread took 73 ms and 2 took 77 ms.
