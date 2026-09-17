# Changelog — comfyui-hicache

All notable changes, per version. Auto-generated from git tags by
`third_party/launch_materials/gen_changelogs.sh`; do not edit by hand.

## v0.2.4 — 2026-09-17

- Force-reinstall hicache-pp from git master in CI (same version as PyPI) (340272a)
- Install hicache-pp from git master in CI (tests need unreleased telemetry API) (32e619e)
- Enforce LF line endings with .gitattributes (43615cc)
- docs: replace interim README visuals (6be05f1)
- docs: refresh README visuals (1ff5da3)
- wip: release readiness pass (a65aa6a)
- build: modernize package license metadata (ee51314)
- ci: update actions to Node 24 runtimes (e5edb20)
- fix: publish registry releases only from version tags (a6f5a9b)
- fix: make clean package builds reproducible (22c9765)
- fix: handle lazy/GGUF flow models in HiCacheModelPatch (issue #1) (2582dd9)
- README: live Comfy installs badge (4385f7f)
- fix: registry Icon must be a full URL (was relative 'icon.png' -> broken on comfy site) (f3e79a0)
- add keywords (registry tags), bump version (bab6ac8)
- fix registry badge (escape hyphens for shields.io) (1c208c3)
- add icon (registry Icon + README banner), bump version (25fc9f3)
- README: release/license badges (d6f6590)
- ci: auto-publish to Comfy Registry on pyproject version bumps (6f82ae0)
- docs: sort CHANGELOG under version headers (df78064)

## v0.2.0 — 2026-06-11

- docs: add per-version CHANGELOG (5e6ec0d)
- Remove REGISTRY-SUBMIT.md (publishing handled directly via comfy-cli) (cf50b65)
- v0.2.0: GPU-validated inside ComfyUI; copy-on-patch cache safety; single-step reset fix (ebe139e)
- ComfyUI-HiCache v0.1.0 (beta): training-free Hunyuan3D acceleration node (26615c4)

