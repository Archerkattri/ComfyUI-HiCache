"""Patch-logic tests on a mock DiT + a synthetic denoise loop (CPU only).

The mock pipeline reproduces the exact call shape of the Hunyuan3D wrapper's
flow-matching loop (kijai/ComfyUI-Hunyuan3DWrapper, hy3dgen/shapegen and
hy3dshape pipelines):

    latent_model_input = cat([latents] * 2)          # batched CFG
    timestep = t.expand(2B) / num_train_timesteps    # increasing, starts at 0
    noise_pred = self.model(latent_model_input, timestep, cond, guidance=None)
    cond_pred, uncond_pred = noise_pred.chunk(2)
    noise_pred = uncond_pred + scale * (cond_pred - uncond_pred)

These tests assert that with the HiCacheModelPatch installed:
  * the inner DiT is called exactly on the scheduled compute steps,
  * skipped steps are filled by the forecaster (finite, accurate on smooth
    velocity trajectories),
  * the per-method forecast path (hermite / dmd / auto) is actually exercised,
  * state resets across sampling runs,
  * parameter plumbing and validation behave.
"""
import importlib.util
import math
import pathlib
import sys

import pytest
import torch

PACK_DIR = pathlib.Path(__file__).resolve().parents[1]


def _load(modname):
    spec = importlib.util.spec_from_file_location(
        f"hicache_patch_under_test_{modname}", PACK_DIR / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hp = _load("hicache_patch")


class MockDiT(torch.nn.Module):
    """Stands in for the Hunyuan3D DiT: same call signature as the wrapper's
    ``self.model(latent_model_input, timestep, cond, guidance=...)``; returns a
    smooth, timestep-dependent velocity field and counts real forwards."""

    guidance_embed = False  # the pipelines hasattr/check this on .model

    def __init__(self, shape=(2, 8, 4)):
        super().__init__()
        self.shape = shape
        self.calls = 0
        self.call_timesteps = []
        base = torch.randn(shape)
        self.register_buffer("base", base)

    def forward(self, latent_model_input, timestep, cond, guidance=None):
        self.calls += 1
        t = float(timestep.reshape(-1)[0])
        self.call_timesteps.append(t)
        # smooth damped-oscillatory velocity trajectory (the feature-ODE class)
        return self.base * math.exp(-1.5 * t) * math.cos(3.0 * t) + 0.3 * t


class MockPipeline:
    """Minimal stand-in for Hunyuan3DDiTFlowMatchingPipeline's denoise loop."""

    num_train_timesteps = 1000

    def __init__(self, model):
        self.model = model

    def __call__(self, num_inference_steps=30, guidance_scale=5.5):
        sigmas = torch.linspace(0, 1, num_inference_steps)
        timesteps = sigmas * self.num_train_timesteps  # increasing, starts at 0
        latents = torch.randn(1, 8, 4)
        outputs = []
        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0]) / self.num_train_timesteps
            noise_pred = self.model(latent_model_input, timestep, cond=None, guidance=None)
            cond_pred, uncond_pred = noise_pred.chunk(2)
            noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            latents = latents - 0.01 * noise_pred  # toy scheduler step
            outputs.append(noise_pred.clone())
        return latents, outputs


def expected_compute_steps(num_steps, interval, warmup):
    """Reference schedule: hicache_pp computes warmup steps, then 1 of every
    `interval` (counter-based: a compute step resets the skip counter). The
    patch floors warmup at 1 — the first step of a run has nothing to forecast
    from."""
    warmup = max(1, warmup)
    computes, counter = [], 0
    for s in range(num_steps):
        if s < warmup or counter >= interval - 1:
            computes.append(s)
            counter = 0
        else:
            counter += 1
    return computes


# ---------------------------------------------------------------------------
# schedule / skip behaviour
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["hermite", "dmd", "auto"])
@pytest.mark.parametrize("interval,warmup", [(3, 2), (5, 2), (4, 0)])
def test_compute_calls_follow_schedule(method, interval, warmup):
    torch.manual_seed(0)
    steps = 30
    dit = MockDiT()
    pipe = MockPipeline(dit)
    pipe = hp.apply_hicache(pipe, method=method, interval=interval, warmup_steps=warmup)

    pipe(num_inference_steps=steps)

    expected = expected_compute_steps(steps, interval, warmup)
    assert dit.calls == len(expected), (
        f"DiT ran {dit.calls} times, schedule says {len(expected)}")
    assert pipe.model.computed_steps == len(expected)
    assert pipe.model.skipped_steps == steps - len(expected)
    # the real forwards happened exactly at the scheduled timesteps
    sigmas = torch.linspace(0, 1, steps)
    assert dit.call_timesteps == pytest.approx([float(sigmas[i]) for i in expected])


class ConstantDiT(MockDiT):
    """Constant velocity — every forecast basis is exact on this class
    (all finite differences vanish), independent of sign/scale conventions."""

    def forward(self, latent_model_input, timestep, cond, guidance=None):
        self.calls += 1
        t = float(timestep.reshape(-1)[0])
        self.call_timesteps.append(t)
        return self.base.clone()


class ExpDiT(MockDiT):
    """Velocity exponential in the timestep: base * exp(-a*t) — the DMD/Prony
    solution class, on which the DMD forecast is exact once its uniform
    snapshot window (4) has filled."""

    def forward(self, latent_model_input, timestep, cond, guidance=None):
        self.calls += 1
        t = float(timestep.reshape(-1)[0])
        self.call_timesteps.append(t)
        return self.base * math.exp(-2.0 * t)


def _run_pair(ref_dit, acc_dit, method, steps=30, interval=4, warmup=2, **kw):
    acc_dit.load_state_dict(ref_dit.state_dict())
    torch.manual_seed(1)
    _, ref_outputs = MockPipeline(ref_dit)(num_inference_steps=steps)
    pipe = MockPipeline(acc_dit)
    pipe = hp.apply_hicache(pipe, method=method, interval=interval, warmup_steps=warmup, **kw)
    torch.manual_seed(1)
    _, acc_outputs = pipe(num_inference_steps=steps)
    computes = expected_compute_steps(steps, interval, warmup)
    skipped = [s for s in range(steps) if s not in computes]
    assert skipped, "test must exercise skipped steps"
    return ref_outputs, acc_outputs, computes, skipped


@pytest.mark.parametrize("method", ["hermite", "dmd", "auto"])
def test_forecast_exact_on_constant_trajectory(method):
    """Convention-free exactness check: on a constant velocity every skipped
    step must reproduce the real DiT output bit-for-bit (all finite
    differences vanish; the cached anchor is the exact forecast)."""
    torch.manual_seed(0)
    ref_outputs, acc_outputs, _, skipped = _run_pair(
        ConstantDiT(), ConstantDiT(), method)
    for s in skipped:
        assert torch.allclose(acc_outputs[s], ref_outputs[s]), \
            f"step {s}: forecast not exact on constant series"


def test_patch_reproduces_hicache_pp_forecast_bitwise():
    """The node's claim is that it faithfully ports the hunyuan2-plus-plus
    pipeline wiring of hicache-pp. Replay the patched run's compute anchors
    through hicache_pp's own state machine and check that the patch served
    the library's forecast bit-for-bit on every skipped step."""
    from hicache_pp import (hicache_init, hicache_decide,
                            hicache_update_derivatives, hicache_forecast)

    torch.manual_seed(0)
    steps, interval, warmup = 30, 4, 2
    dit_a = MockDiT()
    dit_b = MockDiT()
    dit_b.load_state_dict(dit_a.state_dict())

    pipe = MockPipeline(dit_a)
    pipe = hp.apply_hicache(pipe, method="hermite", interval=interval, warmup_steps=warmup)
    torch.manual_seed(1)
    _, acc_outputs = pipe(num_inference_steps=steps)

    # reference: drive hicache_pp directly, exactly like the adapter's loop
    state = hicache_init(num_steps=steps, interval=interval, max_order=1,
                         first_enhance=warmup, end_enhance=steps + 1, sigma=0.5)
    sigmas = torch.linspace(0, 1, steps)
    latents = torch.randn(1, 8, 4)  # same shapes; model output ignores latents
    for s in range(steps):
        if hicache_decide(state) == "forecast":
            ref_raw = hicache_forecast(state)
        else:
            lmi = torch.cat([latents] * 2)
            timestep = sigmas[s].expand(lmi.shape[0])
            ref_raw = dit_b(lmi, timestep, cond=None)
            hicache_update_derivatives(state, ref_raw.detach())
        state["step"] += 1
        # same CFG combine the pipeline loop applies after the model call
        cond_pred, uncond_pred = ref_raw.chunk(2)
        ref = uncond_pred + 5.5 * (cond_pred - uncond_pred)
        assert torch.equal(acc_outputs[s], ref), \
            f"step {s}: patch output diverges from direct hicache_pp wiring"


def test_dmd_forecast_exact_on_exponential_trajectory():
    """DMD is exact on its solution class (sums of exponentials) once 4
    uniformly spaced snapshots exist; before that it falls back to Hermite
    (which only needs to stay finite here)."""
    torch.manual_seed(0)
    steps, interval, warmup = 30, 4, 2
    ref_outputs, acc_outputs, computes, skipped = _run_pair(
        ExpDiT(), ExpDiT(), "dmd", steps=steps, interval=interval, warmup=warmup)
    # 4th uniformly spaced compute step (spacing=interval) closes the window
    uniform = [c for c in computes if c >= warmup]
    window_full_at = uniform[3]
    checked = 0
    for s in skipped:
        assert torch.isfinite(acc_outputs[s]).all()
        if s > window_full_at:
            rel = float((acc_outputs[s] - ref_outputs[s]).norm()
                        / ref_outputs[s].norm().clamp_min(1e-9))
            assert rel < 1e-3, f"step {s}: DMD not exact on exponential ({rel})"
            checked += 1
    assert checked >= 6, "too few post-window skipped steps exercised"


def test_skipped_outputs_finite_and_better_than_naive_reuse():
    """On a smooth exponential trajectory the DMD forecast must be finite on
    every skipped step and, in aggregate, beat the naive 'reuse the last
    computed output' cache (what plain output caching would do)."""
    torch.manual_seed(0)
    steps, interval, warmup = 30, 4, 2
    ref_outputs, acc_outputs, computes, skipped = _run_pair(
        ExpDiT(), ExpDiT(), "dmd", steps=steps, interval=interval, warmup=warmup)

    # every skipped step must be forecast-filled with a finite tensor
    for s in skipped:
        assert torch.isfinite(acc_outputs[s]).all()

    # once the DMD snapshot window has filled (4 uniformly spaced computes),
    # the forecast must beat the naive 'reuse last computed output' cache
    window_full_at = [c for c in computes if c >= warmup][3]
    forecast_err, reuse_err = 0.0, 0.0
    for s in (s for s in skipped if s > window_full_at):
        last_compute = max(c for c in computes if c < s)
        forecast_err += float((acc_outputs[s] - ref_outputs[s]).norm())
        reuse_err += float((ref_outputs[last_compute] - ref_outputs[s]).norm())
    assert reuse_err > 0
    assert forecast_err < 0.1 * reuse_err, (
        f"forecast ({forecast_err:.4f}) must beat naive reuse ({reuse_err:.4f})")

    # compute steps are bit-identical to the reference model output
    for s in computes:
        assert torch.allclose(acc_outputs[s], ref_outputs[s])


def test_interval_1_or_warmup_dominates_means_no_skips():
    dit = MockDiT()
    pipe = MockPipeline(dit)
    pipe = hp.apply_hicache(pipe, method="hermite", interval=2, warmup_steps=100)
    pipe(num_inference_steps=20)
    assert dit.calls == 20 and pipe.model.skipped_steps == 0


# ---------------------------------------------------------------------------
# method plumbing: the right forecaster actually runs
# ---------------------------------------------------------------------------
def test_dmd_method_populates_snapshots_and_forecasts():
    dit = MockDiT()
    pipe = MockPipeline(dit)
    pipe = hp.apply_hicache(pipe, method="dmd", interval=4, warmup_steps=2, dmd_history=5)
    pipe(num_inference_steps=40)
    st = pipe.model._state
    assert st["backend"] == "dmd"
    assert len(st["dmd_snapshots"]) >= 4, "DMD snapshot window never filled"
    assert pipe.model.skipped_steps > 0


def test_hermite_method_keeps_snapshots_empty():
    dit = MockDiT()
    pipe = MockPipeline(dit)
    pipe = hp.apply_hicache(pipe, method="hermite", interval=4, warmup_steps=2)
    pipe(num_inference_steps=40)
    st = pipe.model._state
    assert st["backend"] == "hermite"
    assert st["dmd_snapshots"] == []
    assert st["derivatives"], "Hermite anchors missing"


def test_auto_method_makes_a_holdout_choice():
    dit = MockDiT()
    pipe = MockPipeline(dit)
    pipe = hp.apply_hicache(pipe, method="auto", interval=3, warmup_steps=2, dmd_history=8)
    pipe(num_inference_steps=40)
    st = pipe.model._state
    assert st["backend"] == "auto"
    # auto caches its per-compute-step holdout selection once enough snapshots exist
    assert st.get("_auto_choice") in ("dmd", "hermite")


# ---------------------------------------------------------------------------
# run-boundary reset + passthrough
# ---------------------------------------------------------------------------
def test_state_resets_between_sampling_runs():
    dit = MockDiT()
    pipe = MockPipeline(dit)
    pipe = hp.apply_hicache(pipe, method="dmd", interval=4, warmup_steps=2)

    pipe(num_inference_steps=20)
    first_run_calls = dit.calls
    pipe(num_inference_steps=20)  # timestep drops back to 0 -> reset

    expected = len(expected_compute_steps(20, 4, 2))
    assert first_run_calls == expected
    assert dit.calls == 2 * expected, "second run must re-warm, not inherit state"
    assert pipe.model.computed_steps == expected  # stats are per-run


def test_attribute_and_device_passthrough():
    dit = MockDiT()
    pipe = MockPipeline(dit)
    pipe = hp.apply_hicache(pipe, method="hermite", interval=4, warmup_steps=2)
    # pipelines check hasattr(self.model, 'guidance_embed') and call .to()
    assert pipe.model.guidance_embed is False
    pipe.model.to(torch.float32)
    assert pipe.model.inner is dit


def test_wrap_none_does_not_crash_on_attr_access():
    """Regression: lazy / GGUF pipelines may hand the wrapper a not-yet-loaded
    (None) model. Wrapping None must not raise, and an unknown-attribute lookup
    must give a clear 'not loaded yet' error, NOT the misleading
    'HiCacheModelPatch object has no attribute inner'."""
    patch = hp.HiCacheModelPatch(None, method="hermite", interval=3, warmup_steps=2)
    assert patch.inner is None
    with pytest.raises(AttributeError) as ei:
        _ = patch.guidance_embed          # forwarded lookup on a None inner
    msg = str(ei.value)
    assert "not loaded yet" in msg and "guidance_embed" in msg
    with pytest.raises(RuntimeError):     # a compute step on an unbound patch
        patch(torch.zeros(2, 8, 4), torch.tensor([0.0, 0.0]), cond=None)


def test_bind_inner_materializes_lazy_model():
    """After the real DiT loads, bind_inner attaches it (as a submodule) and the
    patch runs/skip-forecasts through the denoise loop."""
    patch = hp.HiCacheModelPatch(None, method="hermite", interval=3, warmup_steps=2)
    dit = MockDiT()
    assert patch.bind_inner(dit) is patch
    assert patch.inner is dit
    assert "inner" in patch._modules              # registered submodule
    assert patch.guidance_embed is False          # passthrough works now
    steps = 20
    for t in torch.linspace(0, 1, steps):
        patch(torch.zeros(2, 8, 4), t.expand(2), cond=None)
    assert patch.skipped_steps > 0
    assert dit.calls == patch.computed_steps


def test_eager_inner_still_registered_as_submodule():
    """Eager path unbroken: a real nn.Module inner stays a registered submodule so
    state_dict / parameters / device moves recurse into it."""
    dit = MockDiT()
    patch = hp.HiCacheModelPatch(dit, method="hermite", interval=3)
    assert patch.inner is dit
    assert "inner" in patch._modules
    assert any(k.startswith("inner.") for k in patch.state_dict().keys())


def test_remove_restores_original_model():
    dit = MockDiT()
    pipe = MockPipeline(dit)
    patched = hp.apply_hicache(pipe, method="dmd", interval=4, warmup_steps=2)
    assert patched is not pipe and patched.model is not dit
    assert pipe.model is dit, "apply must not mutate its input pipeline"
    restored = hp.remove_hicache(patched)
    assert restored.model is dit
    assert patched.model is not dit, "remove must not mutate its input either"
    assert hp.remove_hicache(pipe) is pipe  # unpatched input passes through


def test_apply_does_not_alias_between_configs():
    """ComfyUI cache-aliasing regression (found in GPU validation): node
    outputs are cached keyed on node inputs, so a pipeline returned for
    config A must keep running config A even after the same loader pipeline
    is later patched with config B. In-place patching broke this: a cached
    'hermite interval=3' output silently ran 'dmd interval=5'."""
    torch.manual_seed(0)
    dit = MockDiT()
    base = MockPipeline(dit)

    pipe_a = hp.apply_hicache(base, method="hermite", interval=3, warmup_steps=2)
    pipe_b = hp.apply_hicache(base, method="dmd", interval=5, warmup_steps=2)

    # the loader's pipeline is untouched; each copy owns its config
    assert base.model is dit
    assert (pipe_a.model.method, pipe_a.model.interval) == ("hermite", 3)
    assert (pipe_b.model.method, pipe_b.model.interval) == ("dmd", 5)
    assert pipe_a.model.inner is dit and pipe_b.model.inner is dit

    # serving the A copy after B was created still runs A's schedule
    steps = 30
    pipe_a(num_inference_steps=steps)
    assert dit.calls == len(expected_compute_steps(steps, 3, 2))


def test_apply_on_patched_pipeline_replaces_not_nests():
    dit = MockDiT()
    base = MockPipeline(dit)
    once = hp.apply_hicache(base, method="hermite", interval=3, warmup_steps=2)
    twice = hp.apply_hicache(once, method="dmd", interval=5, warmup_steps=2)
    assert twice.model.inner is dit, "patch must replace, never nest"
    assert (once.model.method, once.model.interval) == ("hermite", 3)


def test_single_step_runs_reset_at_equal_timestep():
    """Run-boundary regression: two back-to-back single-step runs both call
    the model at t=0. The boundary check must treat the *equal* timestep as
    a new run and recompute — not serve the previous run's cached anchor."""
    dit = MockDiT()
    patch = hp.HiCacheModelPatch(dit, method="hermite", interval=3,
                                 warmup_steps=1)
    lmi = torch.randn(2, 8, 4)
    t0 = torch.zeros(2)

    out1 = patch(lmi, t0, cond=None)            # run 1 (single step)
    with torch.no_grad():
        dit.base.add_(1.0)                      # "new image": DiT output changes
    out2 = patch(lmi, t0, cond=None)            # run 2 (single step, same t)

    assert dit.calls == 2, "second single-step run must recompute, not forecast"
    assert not torch.allclose(out1, out2), "stale anchor served across runs"
    assert patch.computed_steps == 1 and patch.skipped_steps == 0  # per-run stats


# ---------------------------------------------------------------------------
# parameter validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs,match", [
    (dict(method="taylor"), "method"),
    (dict(interval=0), "interval"),
    (dict(warmup_steps=-1), "warmup_steps"),
    (dict(max_order=0), "max_order"),
    (dict(sigma=1.0), "sigma"),
    (dict(sigma=0.0), "sigma"),
    (dict(dmd_history=2), "dmd_history"),
])
def test_bad_params_rejected(kwargs, match):
    base = dict(method="dmd", interval=4, warmup_steps=2,
                max_order=1, sigma=0.5, dmd_history=5)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        hp.validate_config(**base)


def test_apply_requires_model_attribute():
    with pytest.raises(TypeError, match="no `.model` attribute"):
        hp.apply_hicache(object(), method="dmd")
