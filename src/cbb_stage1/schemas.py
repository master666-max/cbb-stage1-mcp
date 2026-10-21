"""Where the box schemas live.

The project grew up with the schemas in a *sibling* design tree —
``<parent>/设计/抽取段-v6/schemas`` — which is fine inside one workspace and fatal for a
published package: a clone has no sibling, so the box writer refuses to build and
``field_coverage`` has nothing to check against.

So the resolver walks a chain:

1. ``<package parent>/schemas`` — the published layout (schemas shipped with the code);
2. ``<project>/schemas`` — a checkout that keeps them at the root;
3. the legacy design-tree sibling — the in-repo layout this project was built in.

Nothing is guessed silently: :func:`schemas_dir` raises with the whole chain when none of
them exists, so a broken install says which paths it looked at instead of reporting a
missing file somewhere downstream.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["candidate_dirs", "find_script", "schemas_dir"]

#: The legacy in-workspace location, relative to the project root.
LEGACY_PREFIX = ("设计", "抽取段-v6")
LEGACY = LEGACY_PREFIX + ("schemas",)


def candidate_dirs(start: str | Path | None = None) -> tuple[Path, ...]:
    """Every place the schemas may live, most-preferred first."""
    if start is None:
        # ``src/cbb_stage1/schemas.py`` -> ``src/cbb_stage1`` -> ``src`` -> project
        here = Path(__file__).resolve()
        package = here.parent
        project = package.parents[1]
    else:
        project = Path(start).resolve()
        package = project / "src" / "cbb_stage1"

    return (
        project / "schemas",
        package / "schemas",
        project.parent.joinpath(*LEGACY),
        project.parents[1].joinpath(*LEGACY) if len(project.parents) > 1 else project / "schemas",
    )


def find_script(name: str, start: str | Path | None = None) -> Path | None:
    """Locate a design-tree script that ships with the code (``box_lint.py``).

    Same defect as the schemas, found the same way: the canonical lint is one of the box
    writer's hard gates, and ``box/lint.py`` went straight to the sibling design tree —
    so every build in a published checkout failed with
    ``BuildError: canonical box_lint gate failed``. Measured on the cleaned publish tree:
    50 tests failed for exactly this.

    Returns ``None`` rather than raising: the caller knows what a missing lint means and
    should say it in its own words.
    """
    if start is None:
        here = Path(__file__).resolve()
        project = here.parents[1]
        package = here.parent
    else:
        project = Path(start).resolve()
        package = project / "src" / "cbb_stage1"

    candidates = (
        project / "tools" / name,
        package / "tools" / name,
        package / "box" / name,
        project.parent.joinpath(*LEGACY_PREFIX, "tools", name),
        project.parents[1].joinpath(*LEGACY_PREFIX, "tools", name)
        if len(project.parents) > 1
        else project / "tools" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def schemas_dir(start: str | Path | None = None) -> Path:
    """Return the first candidate that actually holds schema files."""
    tried: list[Path] = []
    for candidate in candidate_dirs(start):
        tried.append(candidate)
        if candidate.is_dir() and any(candidate.glob("*.schema.json")):
            return candidate
    listing = "\n".join(f"  - {path}" for path in tried)
    raise FileNotFoundError(
        "box schemas not found. Looked in:\n"
        f"{listing}\n"
        "Ship the schemas next to the code (``schemas/``), or point the caller at them "
        "with an explicit schema_dir."
    )
