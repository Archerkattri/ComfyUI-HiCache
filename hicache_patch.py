"""HiCache / HiCache++ model-forward patch — pure Python, no ComfyUI imports.

This module contains all of the actual acceleration logic so it can be unit
tested standalone (no ComfyUI install needed). ``nodes.py`` only does the
ComfyUI plumbing around :func:`apply_hicache` / :func:`remove_hicache`.

How it works
------------
The Hunyuan3D shape pipelines (both the 2.0 ``hy3dgen`` and 2.1 ``hy3dshape``
variants vendored by kijai/ComfyUI-Hunyuan3DWrapper, and Tencent's upstream
``Hunyuan3DDiTFlowMatchingPipeline``) run a flow-matching denoise loop of the
form::

    for t in timesteps:                                  # sigma goes 0 -> 1
        latent_model_input = cat([latents] * 2)          # batched CFG
        noise_pred = self.model(latent_model_input, timestep, cond, ...)
        ...chunk(2) + CFG combine + scheduler.step...

``self.model`` is a plain instance attribute (the DiT ``nn.Module``), so
replacing ``pipeline.model`` with :class:`HiCacheModelPatch` intercepts every
DiT forward. On *compute* steps the wrapped DiT runs normally and its raw
output (the stacked cond+uncond velocity) is cached as a forecast anchor; on
*skipped* steps the DiT is **not called at all** — the output is forecast from
the cached anchors with ``hicache-pp``:

* ``hermite`` — HiCache (dual-scaled physicist's Hermite polynomial,
  arXiv:2508.16984).
* ``dmd``     — HiCache++ (Dynamic Mode Decomposition / Prony exponential
  basis; exact on the feature-ODE solution class, so it stays lossless at
  larger skip intervals than the polynomial).
* ``auto``    — holdout-selected per compute step: serve DMD only when it
  demonstrably beats the polynomial on the cached window.

Note on what is cached: the adapter repos (e.g. hunyuan2-plus-plus) wire the
forecast into the pipeline loop and cache the *CFG-combined* velocity. A
model-level patch necessarily caches the *pre-CFG stacked* output instead.
For the Hermite forecaster the two are mathematically identical (the forecast
is linear in the cached features, and CFG combination is linear). For DMD the
fit is done on the stacked trajectory rather than the combined one — in
practice the same dynamics; this is the standard trade-off of a model-level
patch versus first-class pipeline wiring.

Run-boundary detection: these pipelines sample with *strictly increasing*
timesteps within a run (``sigmas = linspace(0, 1, N)``; "we start from 0" in
the upstream source) and call the model exactly once per step. A timestep
value lower than **or equal to** the previous call's therefore marks a new
sampling run, and the forecast state is re-initialised (the equality case
catches back-to-back single-step runs, which both start at t=0 and would
otherwise be served a stale anchor from the previous run). The total step
count is not knowable from inside the model, so the schedule has no
end-of-run always-compute window (``end_enhance``) — only the initial warmup
window.

Cache safety inside ComfyUI: :func:`apply_hicache` / :func:`remove_hicache`
never mutate the pipeline they are given — they return a shallow copy whose
``model`` attribute is replaced (weights are shared, so this costs nothing).
ComfyUI caches node *outputs* keyed on node *inputs*; an in-place patch lets
a cached output alias a pipeline that a later run re-patched with different
settings (GPU-validated failure mode: a cached "hermite interval=3" output
silently ran "dmd interval=5" after an intervening run re-patched the shared
object). With copy-on-patch every cached output permanently owns its own
configuration.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

import torch

from hicache_pp import (
    hicache_init,
    hicache_decide,
    hicache_update_derivatives,
    hicache_forecast,
    dmd_update_snapshots,
    dmd_forecast_state,
    auto_forecast_state,
)

logger = logging.getLogger("ComfyUI-HiCache")

METHODS = ("hermite", "dmd", "auto")

# Sentinel for "unknown total step count": disables the end-of-run
# always-compute window (any real run is far shorter than this).
_NO_END_WINDOW = 1_000_000


def validate_config(method: str, interval: int, warmup_steps: int,
                    max_order: int, sigma: float, dmd_history: int) -> None:
    """Raise ValueError on bad node parameters (mirrors hicache_pp's checks)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if interval < 1:
        raise ValueError(f"interval must be >= 1, got {interval}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if max_order < 1:
        raise ValueError(f"max_order must be >= 1, got {max_order}")
    if not (0.0 < sigma < 1.0):
        raise ValueError(f"sigma must be in (0, 1), got {sigma}")
    if dmd_history < 3:
        raise ValueError(f"dmd_history must be >= 3, got {dmd_history}")


class HiCacheModelPatch(torch.nn.Module):
    """Drop-in replacement for ``pipeline.model`` that skips DiT forwards.

    Wraps the original DiT module; forwards unknown attribute lookups to it so
    pipeline code like ``hasattr(self.model, 'guidance_embed')`` and
    ``self.model.to(device)`` keep working (the inner model is a registered
    submodule, so ``.to()`` / ``.half()`` / state-dict access recurse into it).
    """

    def __init__(self, model: torch.nn.Module, *, method: str = "hermite",
                 interval: int = 3, warmup_steps: int = 2, max_order: int = 1,
                 sigma: float = 0.5, dmd_history: int = 5) -> None:
        validate_config(method, interval, warmup_steps, max_order, sigma, dmd_history)
        super().__init__()
        self.inner = model
        self._hicache_is_patch = True  # marker for apply/remove
        self.method = method
        self.interval = int(interval)
        self.warmup_steps = int(warmup_steps)
        self.max_order = int(max_order)
        self.sigma = float(sigma)
        self.dmd_history = int(dmd_history)

        self._state: Optional[Dict[str, Any]] = None
        self._last_t: Optional[float] = None
        # per-run stats (read by the node / logged at run boundaries)
        self.computed_steps = 0
        self.skipped_steps = 0

    # -- attribute passthrough -------------------------------------------------
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # fall through to the wrapped DiT (guidance_embed, config, dtype, ...)
            inner = super().__getattr__("inner")
            return getattr(inner, name)

    # -- state handling ----------------------------------------------------------
    def _fresh_state(self) -> Dict[str, Any]:
        # hicache_pp validates interval/order/sigma/backend again here.
        return hicache_init(
            num_steps=_NO_END_WINDOW,
            interval=self.interval,
            max_order=self.max_order,
            # the very first step of a run must always compute (there is no
            # anchor to forecast from), so the effective warmup floor is 1
            first_enhance=max(1, self.warmup_steps),
            end_enhance=_NO_END_WINDOW,
            sigma=self.sigma,
            backend=self.method,
            history=self.dmd_history,
        )

    def reset(self) -> None:
        """Drop all cached anchors and per-run stats (new sampling run)."""
        if self._state is not None and (self.computed_steps or self.skipped_steps):
            logger.info(
                "[HiCache] run finished: %d computed + %d skipped DiT steps "
                "(method=%s, interval=%d)",
                self.computed_steps, self.skipped_steps, self.method, self.interval,
            )
        self._state = self._fresh_state()
        self._last_t = None
        self.computed_steps = 0
        self.skipped_steps = 0

    @staticmethod
    def _timestep_value(timestep: Any) -> float:
        if torch.is_tensor(timestep):
            return float(timestep.reshape(-1)[0].item())
        return float(timestep)

    def _forecast(self, state: Dict[str, Any]) -> torch.Tensor:
        if self.method == "dmd":
            return dmd_forecast_state(state)
        if self.method == "auto":
            return auto_forecast_state(state)
        return hicache_forecast(state)

    # -- the patched forward ------------------------------------------------------
    def forward(self, latent_model_input: torch.Tensor, timestep: Any,
                *args: Any, **kwargs: Any) -> torch.Tensor:
        t_val = self._timestep_value(timestep)
        # New sampling run: within a run these pipelines call the model once
        # per step with strictly increasing timesteps (sigma 0 -> 1), so a
        # non-increasing t means the loop restarted. `<=` (not `<`) so that
        # back-to-back single-step runs (t=0 then t=0 again) also reset
        # instead of serving the previous run's anchor.
        if self._state is None or self._last_t is None or t_val <= self._last_t:
            self.reset()
        self._last_t = t_val

        state = self._state
        if hicache_decide(state) == "forecast":
            out = self._forecast(state)
            state["step"] += 1
            self.skipped_steps += 1
            return out

        out = self.inner(latent_model_input, timestep, *args, **kwargs)
        anchor = out.detach()
        hicache_update_derivatives(state, anchor)
        if self.method in ("dmd", "auto"):
            dmd_update_snapshots(state, anchor, state["history"])
        state["step"] += 1
        self.computed_steps += 1
        return out


# ---------------------------------------------------------------------------
# apply / remove on a pipeline object
# ---------------------------------------------------------------------------
def apply_hicache(pipeline: Any, *, method: str = "hermite", interval: int = 3,
                  warmup_steps: int = 2, max_order: int = 1, sigma: float = 0.5,
                  dmd_history: int = 5) -> Any:
    """Return a shallow copy of ``pipeline`` whose ``model`` is patched.

    The input pipeline is NOT mutated (ComfyUI caches node outputs keyed on
    node inputs, so a cached output must own its configuration forever — see
    the module docstring). Weights are shared between the copy and the
    original; only the wrapper object and the ``model`` attribute differ.
    If the given pipeline is already patched, the patch is replaced (never
    nested), so re-running the node with new parameters reconfigures cleanly.
    """
    if not hasattr(pipeline, "model"):
        raise TypeError(
            "HiCache: pipeline has no `.model` attribute - expected a Hunyuan3D "
            f"shape pipeline, got {type(pipeline).__name__}"
        )
    inner = pipeline.model
    if getattr(inner, "_hicache_is_patch", False):
        inner = inner.inner  # replace, never nest
    patched = copy.copy(pipeline)
    patched.model = HiCacheModelPatch(
        inner, method=method, interval=interval, warmup_steps=warmup_steps,
        max_order=max_order, sigma=sigma, dmd_history=dmd_history,
    )
    logger.info(
        "[HiCache] patched copy of %s: method=%s interval=%d warmup=%d",
        type(pipeline).__name__, method, interval, warmup_steps,
    )
    return patched


def remove_hicache(pipeline: Any) -> Any:
    """Return ``pipeline`` with the original DiT on ``.model``.

    If the pipeline is patched, a shallow copy with the patch unwrapped is
    returned (the input is not mutated); if it is not patched, the pipeline
    is returned unchanged.
    """
    model = getattr(pipeline, "model", None)
    if model is not None and getattr(model, "_hicache_is_patch", False):
        clean = copy.copy(pipeline)
        clean.model = model.inner
        logger.info("[HiCache] removed patch from %s", type(pipeline).__name__)
        return clean
    return pipeline
