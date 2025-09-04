# Reserve‑Aware Connectome‑Constrained Hopfield Recall under Tau Propagation

This project is a **research‑style sandbox** I built to explore how memory networks break down when pathology spreads through a brain‑like graph. It explores memory decline in a toy model of Alzheimer’s disease.

It brings together:

* a **synthetic structural connectome** (modular, hemispheric, small‑world, log‑normal weights),
* a **tau‑like propagation process** (diffusion, clearance, amplification, reserve‑aware schedules),
* and a **Hopfield recall system** that degrades through weakening, pruning, silencing, and collapse.


The model is biologically inspired but not biologically exact. The connectome is synthetic (nodes arranged in 2D with hemispheric and modular structure, plus long-range shortcuts), and tau spread is simplified into clearance, thresholded self-amplification, and diffusion via a random-walk Laplacian. Synaptic damage mechanisms, like weakening, pruning, cell death, and silencing, are abstracted but align with broad neuropathological observations. Constraints: no empirical patient data are used, only procedurally generated graphs and patterns. Parameters are tuned for plausibility, not to fit real biomarker curves. The Hopfield network captures associative recall and noise sensitivity but does not represent actual cortical microcircuits.

This model is good for conceptual exploration because it brings together multiple mechanisms, such as network topology, pathology spread, synaptic weakening, and memory recall, into one reproducible framework. It allows you to see how small local changes accumulate into global collapse, how staging emerges from graph distances, and how recall reliability degrades as tau pathology spreads. Because it is lightweight and self-contained, it can be run quickly, modified easily, and used as a sandbox for testing ideas about network degeneration and memory failure.

---

## What this is

This project brings together three components rarely packaged in a single, lightweight script:

1. **Synthetic connectome generation** – a modular, hemispheric, small‑world network with log‑normal edge weights and callosal shortcuts.
2. **Tau‑like pathology dynamics** – diffusion on the connectome, clearance, self‑amplification above thresholds, heterogeneous capacities, and reserve‑aware global schedules.
3. **Hopfield memory recall** – associative patterns stored on the structural backbone, retrieved under progressive weakening, pruning, silencing, and node death.

The result: a synthetic brain that accumulates pathology stage by stage, while memory accuracy and overlap decline. You can watch collapse unfold, quantify staging curves, and test how structural principles shape vulnerability.

**What this isn’t:** a clinical disease model. Nothing here is calibrated to patient data. The point is transparency and tunability: every assumption is explicit, every parameter is in the open, and you can reproduce the whole run in seconds.

---

## TL;DR

* **Single file, NumPy only.** No external data, no hidden dependencies.
* **Deterministic.** RNG seeded, so runs are reproducible.
* **Rich outputs.** Tracks tau spread, stage means, accuracy, overlap, correlation between pathology and recall, and collapse timing.
* **Experiment ready.** Ideal for stress‑testing algorithmic ideas: change diffusion ramps, hub vulnerability, seed placement, or reserve schedules.

---

## Why tau?

Tau is used here as a **generic driver of progressive, spatially structured damage**. It provides a convenient way to couple graph diffusion, local thresholds, and synaptic consequences:

* diffuses over edges (random‑walk Laplacian),
* amplifies once above a small threshold until capped,
* weakens and prunes connections,
* causes temporary silencing or permanent death,
* injects recall noise proportional to burden.

**Biological realism:** limited. But the construction follows recognized network principles:

* distance‑dependent connectivity,
* hemispheric separation with sparse commissural links,
* modular boost of intra‑community edges,
* log‑normal weight distributions,
* centrality‑weighted vulnerability.

This balance makes tau an effective stand‑in for a “burden” field: not realistic data, but structured enough to produce meaningful collapse dynamics.

---

## What you can do with it

* **Probe robustness**: see how associative recall accuracy drops as pathology spreads.
* **Compare conditions**: vary seed locations, diffusion schedules, or centrality weighting.
* **Measure collapse**: extract functional‑collapse times, tau–accuracy correlations, and stage‑by‑stage trajectories.
* **Teach and demo**: clear, reproducible outputs make it suitable for courses on graph diffusion, associative memory, or reserve effects.

---

## Quick start

```bash
python braakhop.py
```

Example output:

```
t=  0 mean=0.001 max=0.030 >0.1=0.00 >0.3=0.00
...
==== SUMMARY (Reserve-aware variant) ====
N (nodes)    : 84
Seeds (idx)  : [4, 57]
Corr(mean_tau, accuracy) over time: -0.933
Functional-collapse time (acc<0.60): 75
```

---

## Example results (default run, RNG seed=0)

* **Nodes:** 84
* **Seeds:** \[4, 57] (one per hemisphere from module 0)
* **Correlation** (mean tau vs. accuracy): -0.933
* **Collapse time:** step 75 (accuracy < 0.60)
* **Stage means** (tau) at selected steps:

  * t=0   -> \[0.030, 0.000, 0.000, 0.000, 0.000, 0.000]
  * t=24  -> \[0.701, 0.007, 0.000, 0.000, 0.000, 0.000]
  * t=60  -> \[1.435, 0.275, 0.002, 0.000, 0.000, 0.000]
  * t=96  -> \[1.640, 1.169, 0.224, 0.000, 0.000, 0.000]
  * t=120 -> \[1.658, 1.405, 0.561, 0.002, 0.000, 0.000]
  * t=149 -> \[1.659, 1.429, 0.941, 0.042, 0.000, 0.000]

---

## Knobs to turn

Every mechanism is parameterized:

* **Connectome:** N, n\_modules, density, lambda\_dist, module\_strength, shortcut\_frac.
* **Seeds & staging:** seed\_amp, seed\_t0, seed\_dur, stage\_edges.
* **Tau dynamics:** alpha\_i, k\_auto\_i, xmax\_i, x\_thresh, diffusion schedule.
* **Pathology → synapses:** a\_weak, k\_prune, p\_del, eta, lam\_c, death\_th, silencing schedule.
* **Recall:** P patterns, rho correlation, steps\_rec, recall noise schedule, flip probability.

---

## Realistic vs. toy

**Grounded in network science:** hemispheres with callosal shortcuts, distance‑decay, modularity, log‑normal weights, diffusion operator.

**Stylized:** parcellation and modules are synthetic, tau kinetics are simplified, and Hopfield recall is a proxy for cognition.

Use it for **comparative experiments** (A vs. B) and insight, not for quantitative prediction.

---

## Reproducibility

* RNG seeded at top of script: `rng = np.random.default_rng(0)`.
* Same Python/NumPy versions -> identical logs.

---

## How to include in your repo

Keep this file as `README.md` in the repo root. Place the script alongside it. Link to modules if you later split them.

---

## License / reuse

Add a license of your choice (MIT, BSD, Apache‑2.0). The code is educational and open for adaptation.

---

## Acknowledgments

This README describes exactly what the script does — no more, no less.
