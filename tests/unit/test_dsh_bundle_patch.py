"""The DSH bundle patch must be a parseable YAML sequence.

This test exists because the bundle shipped with a broken patch and the host refused to
load it:

    failed to parse overlay .../cordis.patch.yml:
    YAMLException: incomplete explicit mapping pair; a key node is missed (1:1)
    1 | :  # DSH bundle patch: register ...

Root cause: the file began with a bare `:` where `- insert:` belonged. YAML read it as
"a mapping key follows", found no key, and stopped at 1:1. The real entry underneath was
indented two spaces, so it never reached the top level either.

The check is structural, not textual: it parses the patch with the same YAML semantics the
host uses (SafeLoader plus a `!!js` passthrough) and asserts the entry is reachable as a
top-level `insert:` item. A file can look right and still be indented wrong; only parsing
catches that.
"""

from __future__ import annotations

from pathlib import Path

import yaml

BUNDLE = Path.home() / ".dsh" / "local-plugins" / "cbb-stage1-mcp"
PATCH = BUNDLE / "cordis.patch.yml"
PROFILE_PATCH = Path.home() / ".dsh" / "profiles" / "desktop" / "cordis.patch.yml"

ENTRY_ID = "cbb-stage1-mcp"
SERVER_NAME = "cbb"


class DshLoader(yaml.SafeLoader):
    """DSH allows `!!js` expressions; plain SafeLoader rejects them."""


DshLoader.add_multi_constructor(
    "tag:yaml.org,2002:", lambda loader, suffix, node: "<tag>"
)


def _load(path: Path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=DshLoader)


# ── the file the host actually reads ─────────────────────────────────────
def test_the_bundle_patch_exists() -> None:
    assert PATCH.is_file(), f"bundle patch missing: {PATCH}"


def test_the_bundle_patch_parses() -> None:
    """The exact failure the host reported: it must parse at all."""
    parsed = _load(PATCH)
    assert parsed is not None, "an empty patch is not a valid patch"


def test_the_bundle_patch_is_a_top_level_sequence() -> None:
    """A DSH patch is a YAML array of entries, not a mapping.

    The broken file parsed as a *mapping attempt* and failed; a file that parsed as a
    mapping would be silently ignored, which is worse than an error.
    """
    parsed = _load(PATCH)
    assert isinstance(parsed, list), f"top level is {type(parsed).__name__}, expected list"


def test_the_entry_is_reachable_as_an_insert_item() -> None:
    """It has to be under `insert:`, which is how a bundle adds a plugin row."""
    parsed = _load(PATCH)
    inserted: list[dict] = []
    for entry in parsed:
        if isinstance(entry, dict) and "insert" in entry:
            inserted.extend(item for item in entry["insert"] if isinstance(item, dict))
    ids = [item.get("id") for item in inserted]
    assert ENTRY_ID in ids, f"{ENTRY_ID} not among inserted ids: {ids}"


def test_the_entry_names_the_mcp_client_with_a_stdio_transport() -> None:
    parsed = _load(PATCH)
    entry = None
    for candidate in parsed:
        if not isinstance(candidate, dict):
            continue
        for item in candidate.get("insert") or ():
            if isinstance(item, dict) and item.get("id") == ENTRY_ID:
                entry = item
    assert entry is not None, "entry not found"
    assert entry.get("name") == "@deepseek-ai/dsh-mcp-client"
    config = entry.get("config") or {}
    assert config.get("serverName") == SERVER_NAME
    assert config.get("transport") == "stdio"
    assert config.get("command"), "a stdio transport needs a command"
    assert config.get("failOnStartupError") is True, (
        "a server that fails to start must report it, not look installed with no tools"
    )


def test_the_declared_command_and_args_exist() -> None:
    """A patch that parses but points at a missing file is the next failure waiting."""
    parsed = _load(PATCH)
    config = None
    for candidate in parsed:
        if not isinstance(candidate, dict):
            continue
        for item in candidate.get("insert") or ():
            if isinstance(item, dict) and item.get("id") == ENTRY_ID:
                config = item.get("config") or {}
    assert config is not None
    command = Path(str(config["command"]))
    assert command.is_file(), f"interpreter not found: {command}"
    args = list(config.get("args") or ())
    server = Path(str(args[-1]))
    assert server.is_file(), f"server entry not found: {server}"


# ── the profile patch must stay valid too ────────────────────────────────
def test_the_profile_patch_still_parses() -> None:
    """A broken profile patch stops the host from starting, not just the bundle."""
    parsed = _load(PROFILE_PATCH)
    assert isinstance(parsed, list)


def test_the_bundle_is_registered_in_the_profile_manifest() -> None:
    import json

    manifest = json.loads(
        (PROFILE_PATCH.parent / "package.json").read_text(encoding="utf-8")
    )
    bundles = manifest["dsh"]["profile"]["bundles"]
    assert "@local/cbb-stage1-mcp" in bundles
    assert "@local/cbb-stage1-mcp" in (manifest.get("dependencies") or {})
