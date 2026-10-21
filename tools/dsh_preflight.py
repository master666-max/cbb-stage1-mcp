"""DSH restart pre-flight: prove the MCP bundle will actually load, not just exist.

Why this exists
---------------
A scratch check reported `SAFE TO RESTART` while the bundle's `cordis.patch.yml` began with
a bare `:` — invalid YAML — and the host refused to load the bundle:

    failed to parse overlay .../cordis.patch.yml:
    YAMLException: incomplete explicit mapping pair; a key node is missed (1:1)

That check parsed the *profile* patch and asserted the bundle patch *existed*. It never
parsed the bundle patch. "The file is there" is not "the file is loadable", and a check
that only ever passes is indistinguishable from no check.

What it verifies, in the host's own terms
-----------------------------------------
The host reads a bundle patch as a **top-level YAML sequence** whose entries are either
`- id: ...` rows or `- insert:` blocks. A mapping parses perfectly and is then silently
ignored — worse than a parse error, because nothing complains. So the checks are
structural: parse it, demand a sequence, and require the entry to be reachable as an
`insert:` item with a stdio transport, an existing command, and an existing server entry.

Usage
-----
    python -X utf8 tools/dsh_preflight.py                 # the real install
    python -X utf8 tools/dsh_preflight.py --json          # machine-readable
    python -X utf8 tools/dsh_preflight.py --bundle X --profile Y   # fixtures

Exit 0 when nothing is an ERROR, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = ["Finding", "check", "main"]

HOME = Path.home()
DEFAULT_PROFILE = HOME / ".dsh" / "profiles" / "desktop"
DEFAULT_BUNDLE = HOME / ".dsh" / "local-plugins" / "cbb-stage1-mcp"
ENTRY_ID = "cbb-stage1-mcp"
BUNDLE_DEP = "@local/cbb-stage1-mcp"
MCP_CLIENT = "@deepseek-ai/dsh-mcp-client"


class DshLoader(yaml.SafeLoader):
    """DSH allows `!!js` expressions; plain SafeLoader rejects them."""


DshLoader.add_multi_constructor(
    "tag:yaml.org,2002:", lambda loader, suffix, node: "<tag>"
)


@dataclass
class Finding:
    check: str
    severity: str  # ERROR | WARN | INFO
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"check": self.check, "severity": self.severity, "detail": self.detail}


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "ERROR"]

    def add(self, check: str, severity: str, detail: str) -> None:
        self.findings.append(Finding(check, severity, detail))


def _load_yaml(path: Path) -> tuple[object | None, str]:
    """Returns (parsed, error_text). Parsing is the point — not existence."""
    if not path.is_file():
        return None, f"missing: {path}"
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=DshLoader), ""
    except Exception as exc:  # noqa: BLE001 - any parse failure is the finding
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"


def _insert_entries(parsed: object) -> list[dict]:
    """Every row reachable through a top-level `insert:` block."""
    rows: list[dict] = []
    if not isinstance(parsed, list):
        return rows
    for entry in parsed:
        if isinstance(entry, dict) and "insert" in entry:
            rows.extend(item for item in entry["insert"] or () if isinstance(item, dict))
    return rows


def check(*, bundle_dir: Path, profile_dir: Path) -> list[Finding]:
    """Inspect an install. Returns findings; ERROR findings block a restart."""
    report = Report()

    # ── 1. the bundle patch must parse, and be a sequence ────────────────
    bundle_patch = bundle_dir / "cordis.patch.yml"
    parsed, error = _load_yaml(bundle_patch)
    if error:
        report.add(
            "bundle_patch_parses",
            "ERROR",
            f"bundle patch is not loadable — {bundle_patch} :: {error}",
        )
        parsed = None
    else:
        report.add("bundle_patch_parses", "INFO", f"parsed {bundle_patch.name}")

    if parsed is not None and not isinstance(parsed, list):
        # A mapping parses and is then ignored: silent, so it is an ERROR not a warning.
        report.add(
            "bundle_patch_is_sequence",
            "ERROR",
            f"top level is {type(parsed).__name__}, expected a list; "
            "a mapping is silently ignored by the loader",
        )

    # ── 2. the entry must be reachable through insert ────────────────────
    entries = _insert_entries(parsed) if parsed is not None else []
    match = next((e for e in entries if e.get("id") == ENTRY_ID), None)
    if match is None:
        report.add(
            "bundle_entry_reachable",
            "ERROR",
            f"{ENTRY_ID} is not reachable as an insert: item"
            + (f" (inserted ids: {[e.get('id') for e in entries]})" if entries else ""),
        )
    else:
        report.add("bundle_entry_reachable", "INFO", f"{ENTRY_ID} found under insert:")

    # ── 3. the entry must describe a usable stdio server ─────────────────
    config: dict = {}
    if match is not None:
        config = match.get("config") or {}
        problems: list[str] = []
        if match.get("name") != MCP_CLIENT:
            problems.append(f"name is {match.get('name')!r}, expected {MCP_CLIENT!r}")
        if config.get("transport") != "stdio":
            problems.append(f"transport is {config.get('transport')!r}, expected 'stdio'")
        if not config.get("command"):
            problems.append("no command (a stdio transport needs one)")
        if config.get("failOnStartupError") is not True:
            problems.append(
                "failOnStartupError is not true — a server that cannot start would "
                "look like a working install with no tools"
            )
        if problems:
            report.add("bundle_entry_shape", "ERROR", "; ".join(problems))
        else:
            report.add("bundle_entry_shape", "INFO", "stdio server described correctly")

    # ── 4. the declared paths must exist ─────────────────────────────────
    if match is not None:
        missing: list[str] = []
        command = config.get("command")
        if command and not Path(str(command)).is_file():
            missing.append(f"command not found: {command}")
        args = list(config.get("args") or ())
        if args:
            server = Path(str(args[-1]))
            if not server.is_file():
                missing.append(f"server entry not found: {server}")
        elif config:
            missing.append("no args: nothing names the server script")
        if missing:
            report.add("bundle_paths_exist", "ERROR", "; ".join(missing))
        else:
            report.add("bundle_paths_exist", "INFO", "command and server entry exist")

    # ── 5. the bundle manifest and the file it points at ─────────────────
    manifest_path = bundle_dir / "package.json"
    if not manifest_path.is_file():
        report.add("bundle_manifest", "ERROR", f"missing: {manifest_path}")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            report.add("bundle_manifest", "ERROR", f"unparseable: {exc}")
        else:
            rel = (manifest.get("dsh") or {}).get("bundle", {}).get("patch")
            if not rel:
                report.add(
                    "bundle_manifest", "ERROR", "no dsh.bundle.patch entry in package.json"
                )
            elif not (bundle_dir / rel).is_file():
                report.add(
                    "bundle_manifest", "ERROR", f"dsh.bundle.patch points at a missing file: {rel}"
                )
            else:
                report.add("bundle_manifest", "INFO", f"bundle patch declared as {rel!r}")

    # ── 6. the profile patch must parse (a broken one stops the host) ────
    profile_patch = profile_dir / "cordis.patch.yml"
    _, profile_error = _load_yaml(profile_patch)
    if profile_error:
        report.add(
            "profile_patch_parses",
            "ERROR",
            f"profile patch is not loadable — {profile_patch} :: {profile_error}",
        )
    else:
        report.add("profile_patch_parses", "INFO", "profile patch parses")

    # ── 7. the profile must register the bundle ──────────────────────────
    profile_manifest = profile_dir / "package.json"
    if not profile_manifest.is_file():
        report.add("profile_registration", "ERROR", f"missing: {profile_manifest}")
    else:
        try:
            profile = json.loads(profile_manifest.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            report.add("profile_registration", "ERROR", f"unparseable: {exc}")
        else:
            bundles = (profile.get("dsh") or {}).get("profile", {}).get("bundles") or []
            deps = profile.get("dependencies") or {}
            problems = []
            if BUNDLE_DEP not in bundles:
                problems.append(f"{BUNDLE_DEP} not in dsh.profile.bundles")
            if BUNDLE_DEP not in deps:
                problems.append(f"{BUNDLE_DEP} not in dependencies")
            if problems:
                report.add("profile_registration", "ERROR", "; ".join(problems))
            else:
                report.add(
                    "profile_registration",
                    "INFO",
                    f"registered ({len(bundles)} bundles, {len(deps)} deps)",
                )

    # ── 8. every patch insert must name something that resolves ──────────
    # Registration and insertion are two separate acts. A package can resolve fine and
    # still have lost its manifest row (that happened: a `pnpm install` rewrote the
    # lockfile, and the next host reconcile pruned the row), or a row can name a package
    # that nothing provides. Only the second stops the row from mounting, so only it is
    # reported -- but it is reported, because a row that mounts nothing looks exactly
    # like a quiet one.
    if not profile_error:
        profile_parsed, _ = _load_yaml(profile_patch)
        if isinstance(profile_parsed, list):
            unresolved: list[str] = []
            for entry in profile_parsed:
                if not isinstance(entry, dict):
                    continue
                for row in entry.get("insert") or ():
                    if not isinstance(row, dict):
                        continue
                    name = str(row.get("name") or "").strip()
                    if not name or name.startswith("@deepseek-ai/"):
                        # Official bundles ship inside the installation, not node_modules,
                        # so their absence from node_modules proves nothing.
                        continue
                    candidates = (
                        profile_dir / "node_modules" / name,
                        profile_dir.parent / "node_modules" / name,
                    )
                    if not any((c / "package.json").is_file() for c in candidates):
                        unresolved.append(f"{row.get('id')}({name})")
            if unresolved:
                report.add(
                    "profile_inserts_resolve",
                    "WARN",
                    "patch inserts a package that resolves nowhere: " + ", ".join(unresolved),
                )
            else:
                report.add(
                    "profile_inserts_resolve", "INFO", "every patch insert resolves"
                )

    return report.findings


def _render(findings: list[Finding]) -> str:
    lines = [
        "=" * 74,
        "DSH BUNDLE PRE-FLIGHT — does this install actually load?",
        "=" * 74,
        "",
    ]
    for finding in findings:
        mark = {"ERROR": "[X]", "WARN": "[!]", "INFO": "[.]"}.get(finding.severity, "[?]")
        lines.append(f"  {mark} {finding.check:26s} {finding.detail}")
    errors = [f for f in findings if f.severity == "ERROR"]
    lines.append("")
    if errors:
        lines.append(f"  ==> DO NOT RESTART — {len(errors)} error(s) above")
    else:
        lines.append("  ==> SAFE TO RESTART")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the DSH MCP bundle will load before restarting the host"
    )
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--json", action="store_true")
    ns = parser.parse_args(argv)

    findings = check(bundle_dir=Path(ns.bundle), profile_dir=Path(ns.profile))

    if ns.json:
        print(
            json.dumps(
                {
                    "ok": not any(f.severity == "ERROR" for f in findings),
                    "findings": [f.as_dict() for f in findings],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(_render(findings))

    return 1 if any(f.severity == "ERROR" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())