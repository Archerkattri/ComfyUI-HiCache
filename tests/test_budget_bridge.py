"""Integration checks for the shared hicache-pp budget contract."""

import importlib.util
import pathlib

import torch


PACK_DIR = pathlib.Path(__file__).resolve().parents[1]


def _load_patch():
    spec = importlib.util.spec_from_file_location("hicache_budget_bridge_patch", PACK_DIR / "hicache_patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("base", torch.ones(1, 2))

    def forward(self, x, timestep, cond, guidance=None):
        return self.base * (1.0 + float(timestep.reshape(-1)[0]))


class _Pipe:
    def __init__(self):
        self.model = _DiT()


def test_budget_cap_forces_full_fallback_and_manifest_counts_actual_decisions():
    patch = _load_patch()
    pipeline = patch.apply_hicache(_Pipe(), method="hermite", interval=4, max_horizon=1)
    for timestep in torch.linspace(0.0, 1.0, 10):
        pipeline.model(torch.ones(1, 2), timestep, None)

    report = pipeline.model.budget_manifest
    assert report["budget"]["max_horizon"] == 1
    assert report["identity"]["stage"] == "shape"
    assert report["counts"]["fallback"] > 0
    assert report["counts"]["full"] + report["counts"]["forecast"] + report["counts"]["fallback"] == 10
    assert all("path" not in event for event in report["events"])

