<p align="center"><img src="icon.png" alt="ComfyUI-HiCache" width="640"></p>

# ComfyUI-HiCache

<p>
  <a href="https://github.com/Archerkattri/ComfyUI-HiCache/releases"><img alt="Release" src="https://img.shields.io/github/v/release/Archerkattri/ComfyUI-HiCache?color=1f6feb"></a>
  <a href="https://registry.comfy.org/nodes/comfyui-hicache"><img alt="Comfy installs" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.comfy.org%2Fnodes%2Fcomfyui-hicache&query=%24.downloads&label=comfy%20installs&color=4b8bbe"></a>
  <a href="https://registry.comfy.org/publishers/archerkattri/nodes/comfyui-hicache"><img alt="Comfy Registry" src="https://img.shields.io/badge/Comfy%20Registry-comfyui--hicache-4b8bbe"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/Archerkattri/ComfyUI-HiCache?color=0d9488"></a>
</p>


**Training-free Hunyuan3D shape-generation acceleration for ComfyUI** — skip DiT
forwards during flow-matching sampling and forecast the velocity instead, using
the [`hicache-pp`](https://pypi.org/project/hicache-pp/) library
([HiCache](https://arxiv.org/abs/2508.16984) Hermite-polynomial and HiCache++
DMD/Prony-exponential forecasters).

> **Status: GPU-validated end-to-end inside ComfyUI** (2026-06-11, RTX 5090,
> real ComfyUI + kijai/ComfyUI-Hunyuan3DWrapper + Hunyuan3D-2mini fp16):
> the patch engages on schedule, sampling runs 2.7x faster at the recommended
> `hermite interval=3`, the accelerated mesh stays well above the
> different-seed noise floor in F-score against the unaccelerated mesh, and
> repeated runs in one session reset state correctly. Measured details in
> *Validated on* below.

## What it does

One node, `HiCache Accelerate (Hunyuan3D)`, wired between
[kijai/ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper)'s
`Hy3DModelLoader` and its mesh sampler (`Hy3DGenerateMesh` /
`Hy3DGenerateMeshMultiView`):

```
Hy3DModelLoader ──pipeline──▶ HiCache Accelerate ──pipeline──▶ Hy3DGenerateMesh
```

The node patches `pipeline.model` (the shape DiT). On *compute* steps the DiT
runs normally and its output is cached as a forecast anchor; on *skipped* steps
the DiT is **not called at all** — the flow-matching velocity is forecast from
the cached anchors. With `interval = N`, roughly `(N-1)/N` of the DiT forwards
are skipped. The patch needs nothing from ComfyUI internals and works on any
pipeline object whose denoise loop calls
`self.model(latent_model_input, timestep, cond, ...)` once per step — which
covers the wrapper's vendored `hy3dgen` (Hunyuan3D 2.0) and `hy3dshape`
(Hunyuan3D 2.1) pipelines, including the mini/turbo checkpoints
([wrapper issue #97](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper/issues/97)
— the patch is checkpoint-agnostic; only the loop shape matters).

### Methods

| method | basis | when to use |
|---|---|---|
| `hermite` *(default)* | HiCache, dual-scaled physicist's Hermite polynomial ([arXiv:2508.16984](https://arxiv.org/abs/2508.16984)) | interval 3; the best-measured setting inside ComfyUI and on both upstream checkpoints |
| `dmd` | HiCache++, Dynamic Mode Decomposition / Prony exponential basis | larger intervals (4 to 5) on the big 2.1 DiT; on mini it measured below the seed-noise floor inside ComfyUI (see *Validated on*) |
| `auto` | holdout-selected per compute step: serves DMD only when it demonstrably beats the polynomial on the cached window | when unsure |

### Inputs

| input | default | meaning |
|---|---|---|
| `pipeline` | — | `HY3DMODEL` from `Hy3DModelLoader` |
| `method` | `hermite` | forecast basis (above) |
| `interval` | 3 | compute 1 step, forecast `interval-1`. `1` disables caching |
| `warmup_steps` | 2 | always compute the first N steps (floored at 1 — the first step has nothing to forecast from) |
| `enable` | true | off = unpatch, restore the original DiT |
| `max_order` | 1 | Hermite / finite-difference order |
| `sigma` | 0.5 | Hermite contraction factor in (0,1) |
| `dmd_history` | 5 | DMD snapshot window length |

Re-running the node with new parameters re-patches cleanly; `enable=false` (or
`interval=1`) restores the original model. State resets automatically at each
new sampling run.

The node never mutates the pipeline it receives: it returns a shallow copy
whose `model` attribute is the patch (weights stay shared, so this costs no
VRAM). This matters because ComfyUI caches node outputs keyed on node inputs;
an in-place patch lets a cached output alias a pipeline that a later run
re-patched with different settings. That failure was actually observed during
GPU validation (a cached `hermite interval=3` output silently ran
`dmd interval=5`) and is covered by a regression test.

## Measured inside ComfyUI (this node, end-to-end)

Setup: ComfyUI 0.24.0, kijai/ComfyUI-Hunyuan3DWrapper master, Hunyuan3D-2mini
fp16 single-file checkpoint, RTX 5090, torch 2.12 cu128. Workflow: LoadImage
(518x518 RGBA crop) -> InvertMask -> Hy3DModelLoader -> HiCacheAccelerate ->
Hy3DGenerateMesh (30 steps, cfg 5.5) -> Hy3DVAEDecode (octree 384) ->
Hy3DExportMesh. Quality metric: F-score@0.05 of the accelerated mesh against
the unaccelerated mesh from the same seed (50k surface samples each, shared
canonical frame, no alignment needed); as a floor, two unaccelerated runs
that differ only in seed score 0.751 on the same metric.

| config | DiT steps run | sampling | sampling speedup | F1@0.05 vs baseline |
|---|---|---:|---:|---:|
| baseline (`enable=false`) | 30/30 | 1.87 s (16.1 it/s) | 1.00x | 1.000 |
| `hermite interval=3` | 11/30 | 0.70 s (43.1 it/s) | **2.68x** | **0.825** |
| `dmd interval=5` | 7/30 | 1.10 s (27.2 it/s) | 1.69x | 0.719 |

Reading: `hermite i3` is comfortably above the 0.751 different-seed floor
(the accelerated mesh is closer to its baseline than a fresh seed is), at a
2.7x sampling speedup. `dmd i5` lands at the floor on mini inside ComfyUI;
its forecast also costs more per skipped step (an SVD), which eats into the
speedup on a DiT this small. Hence the node defaults are `hermite` / `3`.
On mini the end-to-end win is modest in absolute terms (about 1.2 s of a
roughly 9 s run) because VAE decode and marching cubes dominate; the larger
the shape DiT and the higher the step count, the larger the end-to-end share
the node saves.

## Measured numbers (from the adapter repos, outside ComfyUI)

The same forecasters, wired into the upstream Hunyuan3D pipelines (outside
ComfyUI), measured on Toys4K image-to-3D with geometry-preserving A/B
(F-score@0.05 vs. the uncached baseline, solo re-timed speedups). Full tables
and methodology:
[hicache-plus-plus results](https://github.com/Archerkattri/hicache-plus-plus/blob/master/results/RESULTS.md).

**Hunyuan3D-2.1** (baseline F-score 0.911):

| interval | Hermite (HiCache) | DMD (HiCache++) | speedup |
|---:|---:|---:|---:|
| i3 | **0.876** | 0.852 | 1.72× |
| i4 | 0.776 | **0.827** | 1.80× |
| i5 | 0.735 | **0.860** | 1.79× |

**Hunyuan3D-2-mini** (baseline F-score 0.794, 1.89 s/gen): DMD i5 is **exactly
lossless** (0.794) at 1.69 s; HiCache i3 0.792 at 1.58 s.

Source adapter repos (measured, runnable):
[hunyuan2-plus-plus](https://github.com/Archerkattri/hunyuan2-plus-plus) (2.0 + mini),
[hunyuan2.1-plus-plus](https://github.com/Archerkattri/hunyuan2.1-plus-plus) (2.1).

These numbers come from first-class pipeline wiring (the forecast caches the
CFG-combined velocity); the ComfyUI node is a model-level patch that caches
the pre-CFG stacked output instead. For Hermite the two are mathematically
identical. For DMD they are not, and the inside-ComfyUI measurement above
confirms the difference is real on mini: prefer `hermite` in this node unless
you re-A/B `dmd` on your own checkpoint.

## Install

**Via ComfyUI-Manager:** search for `ComfyUI-HiCache` (once indexed), or
*Install via Git URL* → `https://github.com/Archerkattri/ComfyUI-HiCache`.

**Manual:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Archerkattri/ComfyUI-HiCache
pip install -r ComfyUI-HiCache/requirements.txt   # just: hicache-pp
```

Requires [ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper)
to provide the `HY3DMODEL` pipeline the node accelerates.

## Validated on

What has actually been verified, and how:

* **End-to-end GPU validation inside ComfyUI** (2026-06-11, RTX 5090, ComfyUI
  0.24.0, wrapper master, Hunyuan3D-2mini fp16):
  * headless ComfyUI boots with the pack installed, zero import errors, and
    `HiCacheAccelerate` appears in `/object_info`;
  * the full image-to-mesh workflow executes through the ComfyUI API with the
    node wired in, and the patch demonstrably engages: with 30 sampling steps
    the run-boundary log reports `11 computed + 19 skipped` for
    `hermite interval=3` and `7 computed + 23 skipped` for `dmd interval=5`,
    exactly the library schedule, with the sampling rates and meshes in the
    table above;
  * repeated runs in one server session (the same patched pipeline served
    from ComfyUI's node-output cache across prompts) each reset the forecast
    state at the run boundary and produce fresh, seed-dependent meshes;
  * two bugs were found by this validation and are fixed with regression
    tests: (1) ComfyUI cache aliasing, fixed by copy-on-patch (see *Inputs*);
    (2) back-to-back single-step runs both start at t=0, which the old
    strictly-decreasing run-boundary check missed, so the second run could be
    served a stale anchor; the boundary check is now non-increasing.
* **CPU unit tests (38, in `tests/`)**, run `pytest` in this repo, no ComfyUI
  needed:
  * the node pack loads standalone exactly the way ComfyUI loads it
    (`NODE_CLASS_MAPPINGS`, `INPUT_TYPES` schema, zero `comfy` imports);
  * against a mock DiT with the wrapper's exact call signature, driven by a
    synthetic denoise loop replicating the wrapper's flow-matching loop:
    DiT forwards happen **only** on the scheduled compute steps, skipped steps
    are forecast-filled, and the patch's outputs are **bit-identical** to
    driving `hicache-pp`'s state machine directly (the adapter wiring pattern);
  * DMD forecasts are exact (rel. err < 1e-3) on exponential velocity
    trajectories (its solution class) once its snapshot window fills, and
    beat naive last-output reuse by >10x there;
  * run-boundary reset (including the single-step edge), copy-on-patch
    non-aliasing, re-patch/unpatch, attribute & `.to()` passthrough, and
    parameter validation.
* **Adapter-repo GPU measurements**: the identical forecasters and schedule,
  measured in the upstream (non-ComfyUI) Hunyuan3D pipelines (tables above).

**Not yet measured:** the full-size Hunyuan3D 2.0/2.1 checkpoints inside
ComfyUI (the mechanism is identical and checkpoint-agnostic, and the adapter
repos measured those checkpoints upstream, but the in-Comfy numbers above are
mini-only), and the multiview sampler path (`Hy3DGenerateMeshMultiView`).

### One honest design note

The adapter repos wire the forecast into the pipeline loop and cache the
*CFG-combined* velocity; a model-level patch (the only non-invasive option for
a ComfyUI node) caches the *pre-CFG stacked* DiT output instead. For the
Hermite forecaster the two are mathematically identical (forecast and CFG
combine are both linear); for DMD the fit runs on the stacked trajectory
rather than the combined one. The inside-ComfyUI A/B above shows this matters
in practice: DMD's upstream "lossless at interval 5 on mini" result does NOT
transfer to the model-level patch, so this README does not claim it. Hermite
numbers transfer exactly, and that is what the defaults use.

## License

MIT. Not affiliated with Tencent (Hunyuan3D) or kijai (ComfyUI-Hunyuan3DWrapper).
