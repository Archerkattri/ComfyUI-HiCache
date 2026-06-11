"""ComfyUI-HiCache node definitions.

One node, no ComfyUI-internal imports: the patch operates directly on the
pipeline object that kijai/ComfyUI-Hunyuan3DWrapper passes between nodes as
the custom ``HY3DMODEL`` type, so this file imports cleanly outside ComfyUI
too (which is how the unit tests run it).
"""
from __future__ import annotations

try:
    from .hicache_patch import METHODS, apply_hicache, remove_hicache
except ImportError:  # imported outside ComfyUI (tests / tooling), not as a package
    from hicache_patch import METHODS, apply_hicache, remove_hicache


class HiCacheAccelerate:
    """Training-free Hunyuan3D shape-DiT acceleration via velocity forecasting.

    Wire between the Hunyuan3DWrapper model loader (``Hy3DModelLoader``) and
    the mesh sampler (``Hy3DGenerateMesh`` / ``Hy3DGenerateMeshMultiView``).
    On skipped sampling steps the DiT forward is replaced by a cheap forecast
    from cached velocities (HiCache Hermite / HiCache++ DMD), cutting the
    diffusion-sampling wall clock by roughly ``(interval-1)/interval`` of the
    DiT time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HY3DMODEL", {
                    "tooltip": "Hunyuan3D shape pipeline from Hy3DModelLoader "
                               "(kijai/ComfyUI-Hunyuan3DWrapper)."}),
                "method": (list(METHODS), {
                    "default": "hermite",
                    "tooltip": "Forecast basis on skipped steps:\n"
                               "hermite = HiCache (polynomial, arXiv:2508.16984; "
                               "best measured quality inside ComfyUI)\n"
                               "dmd = HiCache++ (exponential / Prony-DMD; worth "
                               "trying at large intervals on the big 2.1 DiT)\n"
                               "auto = holdout-pick the better of the two per "
                               "compute step"}),
                "interval": ("INT", {
                    "default": 3, "min": 1, "max": 12, "step": 1,
                    "tooltip": "Compute one DiT step, then forecast interval-1 "
                               "steps. 1 = caching disabled (every step computed). "
                               "Measured sweet spot inside ComfyUI: hermite "
                               "interval 3."}),
                "warmup_steps": ("INT", {
                    "default": 2, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Always compute the first N sampling steps before "
                               "any forecasting starts."}),
            },
            "optional": {
                "enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off = remove the patch and restore the original "
                               "DiT forward."}),
                "max_order": ("INT", {
                    "default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Highest Hermite / finite-difference order "
                               "(hermite method and warm-up fallback)."}),
                "sigma": ("FLOAT", {
                    "default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05,
                    "tooltip": "Hermite contraction factor in (0,1); keeps "
                               "high-order terms bounded."}),
                "dmd_history": ("INT", {
                    "default": 5, "min": 3, "max": 16, "step": 1,
                    "tooltip": "DMD snapshot window length (dmd/auto methods)."}),
            },
        }

    RETURN_TYPES = ("HY3DMODEL",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "patch"
    CATEGORY = "HiCache"
    DESCRIPTION = (
        "Training-free acceleration of Hunyuan3D shape generation: skip DiT "
        "forwards and forecast the flow-matching velocity instead (HiCache "
        "Hermite / HiCache++ DMD, via the hicache-pp library)."
    )

    def patch(self, pipeline, method="hermite", interval=3, warmup_steps=2,
              enable=True, max_order=1, sigma=0.5, dmd_history=5):
        if not enable or interval <= 1:
            return (remove_hicache(pipeline),)
        return (apply_hicache(
            pipeline,
            method=method,
            interval=interval,
            warmup_steps=warmup_steps,
            max_order=max_order,
            sigma=sigma,
            dmd_history=dmd_history,
        ),)


NODE_CLASS_MAPPINGS = {
    "HiCacheAccelerate": HiCacheAccelerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiCacheAccelerate": "HiCache Accelerate (Hunyuan3D)",
}
