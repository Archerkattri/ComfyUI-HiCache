# Submitting ComfyUI-HiCache to the Comfy Registry

Status: the node is GPU-validated end-to-end (see README, *Validated on*) and
version 0.2.0 is ready to publish. Everything below the "human-only" line is
prepared; the two human-only steps need Krishi's browser login.

## What is already in place

* `pyproject.toml` has the required `[project]` fields (`name`,
  `description`, `version = "0.2.0"`, `license`, `dependencies`) and the
  `[tool.comfy]` block:
  * `PublisherId = "archerkattri"`
  * `DisplayName = "ComfyUI-HiCache"`
* `.comfyignore` excludes tests and tooling files from the published package.
* The repo is public at `https://github.com/Archerkattri/ComfyUI-HiCache`.

## Human-only steps (browser, one time)

1. Create the publisher account: go to https://registry.comfy.org, log in,
   and create a publisher. The publisher ID is the part after the `@` on the
   profile page; it is globally unique and immutable. It MUST be
   `archerkattri` to match `pyproject.toml`. If that ID is taken, pick
   another and update `PublisherId` in `pyproject.toml` before publishing.
2. Create an API key: registry nodes section -> select the publisher ->
   create a new API key. Name it (for example `comfyui-hicache-publish`) and
   store it safely; lost keys cannot be recovered, only replaced.

## Publish (CLI, after the human-only steps)

```bash
pip install comfy-cli            # if not already installed
cd ComfyUI-HiCache               # repo root, where pyproject.toml lives
comfy node publish               # prompts for the API key
```

Expected confirmation: `Version 0.2.0 Published.` The node then appears at
`https://registry.comfy.org/publishers/archerkattri/nodes/comfyui-hicache`
and becomes installable through ComfyUI-Manager once the registry index
picks it up.

Do NOT use the registry's GitHub Actions auto-publish flow
(`publish_action.yml` + `REGISTRY_ACCESS_TOKEN` secret): this repo
deliberately carries no CI. Publish from the CLI for each release.

## Releasing future versions

1. Bump `version` in `pyproject.toml` AND `__version__` in `__init__.py`
   (keep them identical, semver).
2. Run the tests: `pytest` (38 must pass).
3. Commit, push, then `comfy node publish` again with the same API key.
