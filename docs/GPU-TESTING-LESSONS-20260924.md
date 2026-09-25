# GPU testing lessons ? 24 September 2026

The small coordinate microbenchmark was not a ceiling on GPU usefulness. Workload design determines whether the device is fed meaningful parallel work.

## Evidence
The corrected worlds-/night-results/enumerated-total-corrected.json records 661,654,158,300 distinct lattice indices and 31,292,801,587,800 cumulative evaluations. The main overnight log segment beginning 21 September 03:34:27 reconciles to 48 arc/bifacial/hundred-billion sweeps and 47 substation/atlas sweeps; the last pass was partial. Repeated deterministic passes are soak testing, not additional unique coverage.

worlds-/atlas-lattice.json records 295,770,420,000 evaluations in 76.384 seconds. worlds-/night-results/annihilate.json records 5,315,908,358 paired cases / 10,631,816,716 arithmetic evaluations in 2.2773 seconds. These are saved receipts, not reruns performed for this document. Their case definitions and CPU verification scope matter: atlas/substation independently probe 200,000 cases, not the whole space.

## Reusable design
- Generate Cartesian indices on-device rather than building billions of Python objects.
- Keep source arrays and intermediate work in VRAM; aggregate locally, reduce on the GPU, transfer small tallies and bounded counterexamples.
- Fuse operations to remove temporary-array memory traffic. Reuse allocations and compiled kernels. engine/fused.py already demonstrates this architecture.
- Validate the actual application rules, including float precision, strict/inclusive boundaries, nonfinite inputs, disabled layers and zoom thresholds. Independent implementations can share the same wrong assumption.
- Count evaluated combinations, unique input identities, arithmetic operations and repeated passes separately.
- Report load/hash, preparation, transfer/compile, validation, kernel and end-to-end time separately. Synchronize GPU timings. A fast kernel is not an end-to-end speedup by itself.
- Preserve the namespace fix in annihilate.py: worksheet axes previously shadowed kernel variables and created 161 false disagreements.
- Browser integration checks remain necessary. A numeric preflight can cheaply reject many bad states before targeted browser tests; it does not certify event wiring or network delivery.

## First implementation
The companion gpu-drivers-for-global-grid/engine/gridatlas_gpu program starts a real-corpus Cartesian preflight over coordinate rows, captured MapLibre camera states, enabled/disabled states and a screen click lattice. It independently checks projection against the deployed browser, compares a reproducible CPU probe, and tests explicit edge/nonfinite fixtures before the full GPU sweep.

Scope is deliberately explicit: flat Web Mercator coordinate visibility/proximity, no world copies. Coordinates from lines are vertices, not full segment hit tests. The 6px search tolerance is a candidate-selection parameter, not an assertion about MapLibre paint radii. Layer filters, actual point/line radii, occlusion and event delivery are next contracts to extract and verify. Do not inflate the current receipt into those unimplemented claims.

The user-requested GitHub workflow pauses stay in place. Work here is local; no release composition is changed.

## First executed Cartesian preflight
On 24 September, the new gpu-drivers-for-global-grid/engine/gridatlas_gpu/sweep.py evaluated 1,046,601,504 coordinate-row/camera/visibility/click combinations in one pass: 55.32 ms GPU sweep, 1.056 seconds Python pipeline (browser capture separate). CPU parity: 199,976 sampled combinations, zero differences. 144 explicit fixture combinations passed. 960 real browser projections agreed within 0.0000000254 pixels. Receipt and browser capture are stored under that module's results directory. This demonstrates broad numeric enumeration, not a measured speedup of browser CI or full feature-picking correctness.
