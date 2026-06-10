"""Node-pack registration tests — run standalone, no ComfyUI install.

ComfyUI loads a custom node pack by importing the package and reading
``NODE_CLASS_MAPPINGS`` / ``NODE_DISPLAY_NAME_MAPPINGS``; these tests verify
that contract plus the INPUT_TYPES schema shape ComfyUI expects.
"""
import importlib.util
import pathlib
import sys

import pytest

PACK_DIR = pathlib.Path(__file__).resolve().parents[1]


def _load_pack():
    """Import the node pack the way ComfyUI does (as a package by path)."""
    name = "comfyui_hicache_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PACK_DIR / "__init__.py",
        submodule_search_locations=[str(PACK_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pack_imports_without_comfy():
    assert "comfy" not in sys.modules, "test must run without ComfyUI"
    pack = _load_pack()
    assert hasattr(pack, "NODE_CLASS_MAPPINGS")
    assert hasattr(pack, "NODE_DISPLAY_NAME_MAPPINGS")
    assert "comfy" not in sys.modules, "node pack must not import comfy internals"


def test_node_mappings_consistent():
    pack = _load_pack()
    assert "HiCacheAccelerate" in pack.NODE_CLASS_MAPPINGS
    for key in pack.NODE_CLASS_MAPPINGS:
        assert key in pack.NODE_DISPLAY_NAME_MAPPINGS


def test_input_types_schema():
    pack = _load_pack()
    cls = pack.NODE_CLASS_MAPPINGS["HiCacheAccelerate"]
    schema = cls.INPUT_TYPES()
    req = schema["required"]

    # model input + the contract from the task spec: method/interval/warmup
    assert req["pipeline"][0] == "HY3DMODEL"
    assert set(req["method"][0]) == {"hermite", "dmd", "auto"}
    assert req["method"][1]["default"] in req["method"][0]
    assert req["interval"][0] == "INT"
    assert req["interval"][1]["min"] >= 1
    assert req["warmup_steps"][0] == "INT"
    assert req["warmup_steps"][1]["min"] == 0
    assert "enable" in schema["optional"]

    # ComfyUI node contract
    assert cls.RETURN_TYPES == ("HY3DMODEL",)
    assert callable(getattr(cls, cls.FUNCTION))
    assert isinstance(cls.CATEGORY, str)


def test_node_function_patches_and_unpatches():
    import torch

    pack = _load_pack()
    node = pack.NODE_CLASS_MAPPINGS["HiCacheAccelerate"]()

    class FakePipe:
        model = torch.nn.Linear(2, 2)

    pipe = FakePipe()
    original = pipe.model

    (out,) = node.patch(pipe, method="hermite", interval=4, warmup_steps=2)
    assert out is pipe
    assert getattr(pipe.model, "_hicache_is_patch", False)
    assert pipe.model.inner is original

    # re-patch with new params replaces (not nests) the patch
    (out,) = node.patch(pipe, method="dmd", interval=3, warmup_steps=1)
    assert pipe.model.inner is original
    assert pipe.model.method == "dmd" and pipe.model.interval == 3

    # enable=False restores the original model
    (out,) = node.patch(pipe, enable=False)
    assert pipe.model is original

    # interval=1 means "no caching" -> also unpatched
    node.patch(pipe, interval=4)
    (out,) = node.patch(pipe, interval=1)
    assert pipe.model is original


def test_node_rejects_bad_pipeline():
    pack = _load_pack()
    node = pack.NODE_CLASS_MAPPINGS["HiCacheAccelerate"]()
    with pytest.raises(TypeError, match="no `.model` attribute"):
        node.patch(object())
