# ComfyUI-HiCache

**Training-free Hunyuan3D shape-generation acceleration for ComfyUI** — skip DiT
forwards during flow-matching sampling and forecast the velocity instead, using
the [`hicache-pp`](https://pypi.org/project/hicache-pp/) library
([HiCache](https://arxiv.org/abs/2508.16984) Hermite-polynomial and HiCache++
DMD/Prony-exponential forecasters).

> **Status: beta.** The forecasting math, schedule, and patch logic are
> CPU-unit-tested against the library and against the wiring pattern of the
> measured adapter repos (see *Validated on* below). End-to-end inside-ComfyUI
> GPU validation is still pending — until then, treat the node as experimental
> and A/B your own outputs.

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
| `hermite` | HiCache — dual-scaled physicist's Hermite polynomial ([arXiv:2508.16984](https://arxiv.org/abs/2508.16984)) | conservative intervals (i3) |
| `dmd` *(default)* | HiCache++ — Dynamic Mode Decomposition / Prony exponential basis | larger intervals (i4–i5); degrades most gracefully |
| `auto` | holdout-selected per compute step: serves DMD only when it demonstrably beats the polynomial on the cached window | when unsure |

### Inputs

| input | default | meaning |
|---|---|---|
| `pipeline` | — | `HY3DMODEL` from `Hy3DModelLoader` |
| `method` | `dmd` | forecast basis (above) |
| `interval` | 5 | compute 1 step, forecast `interval-1`. `1` disables caching |
| `warmup_steps` | 2 | always compute the first N steps (floored at 1 — the first step has nothing to forecast from) |
| `enable` | true | off = unpatch, restore the original DiT |
| `max_order` | 1 | Hermite / finite-difference order |
| `sigma` | 0.5 | Hermite contraction factor in (0,1) |
| `dmd_history` | 5 | DMD snapshot window length |

Re-running the node with new parameters re-patches cleanly; `enable=false` (or
`interval=1`) restores the original model. State resets automatically at each
new sampling run.

## Measured numbers (from the adapter repos — not yet re-measured inside ComfyUI)

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

These numbers transfer to ComfyUI only insofar as the wrapper's vendored
pipelines run the same denoise loop (they do, structurally — that is what the
unit tests pin down), but **no inside-ComfyUI measurement has been done yet**.

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

* **CPU unit tests (35, in `tests/`)** — run `pytest` in this repo, no ComfyUI
  needed:
  * the node pack loads standalone exactly the way ComfyUI loads it
    (`NODE_CLASS_MAPPINGS`, `INPUT_TYPES` schema, zero `comfy` imports);
  * against a mock DiT with the wrapper's exact call signature, driven by a
    synthetic denoise loop replicating the wrapper's flow-matching loop:
    DiT forwards happen **only** on the scheduled compute steps, skipped steps
    are forecast-filled, and the patch's outputs are **bit-identical** to
    driving `hicache-pp`'s state machine directly (the adapter wiring pattern);
  * DMD forecasts are exact (rel. err < 1e-3) on exponential velocity
    trajectories — its solution class — once its snapshot window fills, and
    beat naive last-output reuse by >10× there;
  * run-boundary reset, re-patch/unpatch, attribute & `.to()` passthrough,
    and parameter validation.
* **Adapter-repo GPU measurements** — the identical forecasters and schedule,
  measured in the upstream (non-ComfyUI) Hunyuan3D pipelines (tables above).

**Not yet validated:** running inside ComfyUI on GPU with the real wrapper and
real checkpoints (including mini/turbo), and quality A/B of the resulting
meshes through the full Comfy graph. Until that lands this repo stays 0.x/beta
and is not submitted to the Comfy Registry.

### One honest design note

The adapter repos wire the forecast into the pipeline loop and cache the
*CFG-combined* velocity; a model-level patch (the only non-invasive option for
a ComfyUI node) caches the *pre-CFG stacked* DiT output instead. For the
Hermite forecaster the two are mathematically identical (forecast and CFG
combine are both linear); for DMD the fit runs on the stacked trajectory
rather than the combined one — same dynamics, but not the literally identical
computation, which is one more reason the GPU A/B is required before any
lossless claim is repeated for this node.

## License

MIT. Not affiliated with Tencent (Hunyuan3D) or kijai (ComfyUI-Hunyuan3DWrapper).
