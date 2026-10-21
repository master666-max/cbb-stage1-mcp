"""The restart pre-flight must catch a broken **bundle** patch, not just the profile one.

Why this file exists
--------------------
A scratch check script reported `SAFE TO RESTART` while the bundle's `cordis.patch.yml`
began with a bare `:` — invalid YAML. The host then refused to load the bundle:

    failed to parse overlay .../cordis.patch.yml:
    YAMLException: incomplete explicit mapping pair; a key node is missed (1:1)

The script parsed the *profile* patch and checked that the bundle patch *existed*, but
never parsed the bundle patch itself. "The file is there" is not "the file is loadable".

These tests are written the way the tool should have been: build a tree, break one thing,
and assert the check reports it. A check that only ever passes is indistinguishable from
no check — which is exactly what happened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from dsh_preflight import check, main  # noqa: E402

GOOD_BUNDLE_PATCH = """\
- insert:
    - id: cbb-stage1-mcp
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: cbb
        transport: stdio
        command: '{command}'
        args:
          - '-u'
          - '{server}'
        failOnStartupError: true
"""

#: The exact defect that shipped: a bare `:` where `- insert:` belonged.
BROKEN_BUNDLE_PATCH = """\
:  # DSH bundle patch: register something
  - id: cbb-stage1-mcp
    name: '@deepseek-ai/dsh-mcp-client'
    config:
      serverName: cbb
      transport: stdio
"""

GOOD_PROFILE_PATCH = """\
- id: some-plugin
  config:
    enabled: true
"""


def _tree(root: Path, *, bundle_patch: str = GOOD_BUNDLE_PATCH) -> dict[str, Path]:
    """A minimal but complete install: bundle dir, profile dir, server entry."""
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    profile = root / "profile"
    profile.mkdir(parents=True)
    server = root / "mcp" / "server.py"
    server.parent.mkdir(parents=True)
    server.write_text("# stand-in for the MCP server entry\n", encoding="utf-8")
    command = Path(sys.executable)

    (bundle / "cordis.patch.yml").write_text(
        bundle_patch.format(command=command.as_posix(), server=server.as_posix()),
        encoding="utf-8",
    )
    (bundle / "package.json").write_text(
        json.dumps(
            {
                "name": "@local/cbb-stage1-mcp",
                "dsh": {"bundle": {"patch": "./cordis.patch.yml"}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (profile / "cordis.patch.yml").write_text(GOOD_PROFILE_PATCH, encoding="utf-8")
    (profile / "package.json").write_text(
        json.dumps(
            {
                "dependencies": {"@local/cbb-stage1-mcp": f"file:{bundle.as_posix()}"},
                "dsh": {"profile": {"bundles": ["@local/cbb-stage1-mcp"]}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"bundle": bundle, "profile": profile, "server": server}


# ── the healthy case must pass ───────────────────────────────────────────
def test_a_healthy_tree_has_no_errors(tmp_path: Path) -> None:
    paths = _tree(tmp_path)
    findings = check(bundle_dir=paths["bundle"], profile_dir=paths["profile"])
    errors = [f for f in findings if f.severity == "ERROR"]
    assert errors == [], f"unexpected errors: {errors}"


# ── the defect that shipped must be caught ───────────────────────────────
def test_a_bare_colon_patch_is_reported_as_an_error(tmp_path: Path) -> None:
    """The regression test for the actual incident."""
    paths = _tree(tmp_path, bundle_patch=BROKEN_BUNDLE_PATCH)
    findings = check(bundle_dir=paths["bundle"], profile_dir=paths["profile"])
    errors = [f for f in findings if f.severity == "ERROR"]
    assert errors, "an unparseable bundle patch must be an error"
    joined = " ".join(f.detail for f in errors)
    assert "bundle patch" in joined.lower(), f"the message must name the file: {joined}"


def test_the_bundle_patch_is_actually_parsed_not_merely_found(tmp_path: Path) -> None:
    """A file that exists but does not parse must not pass.

    This is the specific gap: the old check asserted `is_file()` and stopped there.
    """
    paths = _tree(tmp_path, bundle_patch=BROKEN_BUNDLE_PATCH)
    findings = check(bundle_dir=paths["bundle"], profile_dir=paths["profile"])
    parsed_findings = [f for f in findings if f.check == "bundle_patch_parses"]
    assert parsed_findings, "there must be a finding about parsing the bundle patch"
    assert parsed_findings[0].severity == "ERROR"


# ── structural defects beyond parse errors ──────────────────────────────
def test_a_patch_that_parses_as_a_mapping_is_an_error(tmp_path: Path) -> None:
    """A mapping parses fine and is silently ignored — worse than a parse error."""
    paths = _tree(tmp_path, bundle_patch="insert: []\nsomething: else\n")
    findings = check(bundle_dir=paths["bundle"], profile_dir=paths["profile"])
    assert any(f.severity == "ERROR" for f in findings)


def test_an_entry_outside_insert_is_an_error(tmp_path: Path) -> None:
    paths = _tree(
        tmp_path,
        bundle_patch=(
            "- id: cbb-stage1-mcp\n"
            "  name: '@deepseek-ai/dsh-mcp-client'\n"
            "  config:\n"
            "    serverName: cbb\n"
            "    transport: stdio\n"
        ),
    )
    findings = check(bundle_dir=paths["bundle"], profile_dir=paths["profile"])
    assert any(
        f.check == "bundle_entry_reachable" and f.severity == "ERROR" for f in findings
    ), f"entry outside insert: must be flagged; got {[f.check for f in findings]}"


def test_a_missing_command_is_an_error(tmp_path: Path) -> None:
    paths = _tree(
        tmp_path,
        bundle_patch=(
            "- insert:\n"
            "    - id: cbb-stage1-mcp\n"
            "      name: '@deepseek-ai/dsh-mcp-client'\n"
            "      config:\n"
            "        serverName: cbb\n"
            "        transport: stdio\n"
            "        failOnStartupError: true\n"
        ),
    )
    findings = check(bundle_dir=paths["bundle"], profile_dir=paths["profile"])
    assert any(f.check == "bundle_entry_shape" and f.severity == "ERROR" for f in findings)


def test_a_command_pointing_at_a_missing_file_is_an_error(tmp_path: Path) -> None:
    paths = _tree(tmp_path)
    patch = paths["bundle"] / "cordis.patch.yml"
    patch.write_text(
        GOOD_BUNDLE_PATCH.format(
            command=(tmp_path / "nope" / "python.exe").as_posix(),
            server=paths["server"].as_posix(),
        ),
        encoding="utf-8",
    )
    findings = check(bundle_dir=paths["bundle"], profile_dir=paths["profile"])
    assert any(f.check == "bundle_paths_exist" and f.severity == "ERROR" for f in findings)


def test_the_bundle_being_absent_from_the_profile_is_an_error(tmp_path: Path) -> None:
    paths = _tree(tmp_path)
    (paths["profile"] / "package.json").write_text(
        json.dumps({"dependencies": {}, "dsh": {"profile": {"bundles": []}}}),
        encoding="utf-8",
    )
    findings = check(bundle_dir=paths["bundle"], profile_dir=paths["profile"])
    assert any(f.check == "profile_registration" and f.severity == "ERROR" for f in findings)


def test_a_broken_profile_patch_is_an_error(tmp_path: Path) -> None:
    """A broken profile patch stops the host from starting, not just the bundle."""
    paths = _tree(tmp_path)
    (paths["profile"] / "cordis.patch.yml").write_text(":  bad\n", encoding="utf-8")
    findings = check(bundle_dir=paths["bundle"], profile_dir=paths["profile"])
    assert any(f.check == "profile_patch_parses" and f.severity == "ERROR" for f in findings)


# ── the exit code is what a shell checks ────────────────────────────────
def test_main_exits_non_zero_on_a_broken_tree(tmp_path: Path, capsys) -> None:
    paths = _tree(tmp_path, bundle_patch=BROKEN_BUNDLE_PATCH)
    code = main([
        "--bundle", str(paths["bundle"]),
        "--profile", str(paths["profile"]),
    ])
    assert code == 1, "a broken install must exit non-zero"


def test_main_exits_zero_on_a_healthy_tree(tmp_path: Path) -> None:
    paths = _tree(tmp_path)
    code = main([
        "--bundle", str(paths["bundle"]),
        "--profile", str(paths["profile"]),
    ])
    assert code == 0


def test_the_output_names_the_single_verdict(tmp_path: Path, capsys) -> None:
    paths = _tree(tmp_path)
    main(["--bundle", str(paths["bundle"]), "--profile", str(paths["profile"])])
    out = capsys.readouterr().out
    assert "SAFE TO RESTART" in out


# ── the patch must not insert something nothing provides ────────────────
def test_an_insert_naming_an_unresolvable_package_is_flagged(tmp_path: Path) -> None:
    """A patch row whose package resolves nowhere is a dangling insert.

    `@deepseek-ai/*` is exempt: those ship inside the installation, not node_modules.
    Anything else has to resolve, or the row mounts nothing.
    """
    paths = _tree(tmp_path)
    (paths["profile"] / "cordis.patch.yml").write_text(
        "- insert:\n"
        "    - id: ghost\n"
        "      name: '@local/does-not-exist'\n"
        "      config:\n"
        "        enabled: true\n",
        encoding="utf-8",
    )
    findings = check(bundle_dir=paths["bundle"], profile_dir=paths["profile"])
    flagged = [
        f
        for f in findings
        if f.check == "profile_inserts_resolve" and f.severity != "INFO"
    ]
    assert flagged, f"an unresolvable insert must be reported; got {[f.check for f in findings]}"
    assert "does-not-exist" in flagged[0].detail, (
        f"the message must name the package; got {flagged[0].detail!r}"
    )


def test_official_bundles_are_not_flagged_as_unresolvable(tmp_path: Path) -> None:
    """`@deepseek-ai/*` lives in the app, so its absence from node_modules is normal."""
    paths = _tree(tmp_path)
    (paths["profile"] / "cordis.patch.yml").write_text(
        "- insert:\n"
        "    - id: base\n"
        "      name: '@deepseek-ai/dsh-base'\n"
        "    - id: hmr\n"
        "      name: '@deepseek-ai/dsh-hmr'\n",
        encoding="utf-8",
    )
    findings = check(bundle_dir=paths["bundle"], profile_dir=paths["profile"])
    ran = [f for f in findings if f.check == "profile_inserts_resolve"]
    assert ran, "the check must have run at all"
    warned = [f for f in ran if f.severity != "INFO"]
    assert warned == [], f"official bundles must be exempt; got {warned}"


def test_a_local_plugin_that_resolves_is_not_flagged(tmp_path: Path) -> None:
    paths = _tree(tmp_path)
    local = paths["profile"] / "node_modules" / "@local" / "present"
    local.mkdir(parents=True)
    (local / "package.json").write_text('{"name": "@local/present"}', encoding="utf-8")
    (paths["profile"] / "cordis.patch.yml").write_text(
        "- insert:\n"
        "    - id: present\n"
        "      name: '@local/present'\n",
        encoding="utf-8",
    )
    findings = check(bundle_dir=paths["bundle"], profile_dir=paths["profile"])
    ran = [f for f in findings if f.check == "profile_inserts_resolve"]
    assert ran, "the check must have run at all"
    warned = [f for f in ran if f.severity != "INFO"]
    assert warned == [], f"a resolvable local plugin must not be flagged; got {warned}"
