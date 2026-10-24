"""A stage list that says only where the code lives cannot answer "may I skip this?".

Measured gap (E8)
-----------------
`orchestrator/postprocess.py:65-84` defines:

    StageSpec(name, module)

so the eight postprocess stages are a call order and nothing more. Three things a reader or a gate
needs are missing:

* **what decides the stage succeeded** — every stage returns `{"status": "ok"}`, but nothing states
  which of its outputs is the thing that must exist;
* **whether it is required** — is a box without `validate` valid? The answer is no, and nothing says so;
* **whether it can be switched off** — the toggles that exist (`config.description_synthesis`,
  `config.summary_synthesis`) are read inside stages, not declared where the stage is listed.

Consequence: the `pass` registry the design asks for does not exist. A reader has to open eight modules
to learn which passes are optional, and a gate cannot know what "this stage ran" means beyond the
status string.

What is pinned
--------------
* every stage declares its required outputs and whether it is required;
* declaring a stage optional without naming its toggle is an error — a switch nobody can find is not a
  switch;
* the registry is the single source: `DEFAULT_STAGES` and the new metadata cannot disagree;
* a `validate`-less box is expressible in principle but the registry marks it as not allowed, so the
  gate has something to enforce.
"""

from __future__ import annotations

from cbb_stage1.orchestrator.postprocess import (
    DEFAULT_STAGES,
    OPTIONAL_STAGE_TOGGLES,
    StageSpec,
    stage_registry,
)


def _by_name() -> dict[str, StageSpec]:
    return {spec.name: spec for spec in DEFAULT_STAGES}


# ══ every stage declares its contract ════════════════════════════════
def test_every_stage_declares_its_required_outputs() -> None:
    for spec in DEFAULT_STAGES:
        assert spec.required_outputs, f"{spec.name} declares no required outputs"
        assert all(isinstance(item, str) and item for item in spec.required_outputs)


def test_every_stage_says_whether_it_is_required() -> None:
    for spec in DEFAULT_STAGES:
        assert isinstance(spec.required, bool), f"{spec.name} does not say whether it is required"


def test_every_stage_states_its_criterion() -> None:
    for spec in DEFAULT_STAGES:
        assert spec.criterion.strip(), f"{spec.name} has no success criterion"


# ══ the required set is the one the box actually needs ═══════════════
def test_validate_is_required() -> None:
    """A box with unvalidated assertions is not a valid box."""
    assert _by_name()["validate"].required is True


def test_derive_is_optional() -> None:
    """Derived tables are useful and not load-bearing for the knowledge layer."""
    assert _by_name()["derive"].required is False


def test_resolve_and_assemble_are_required() -> None:
    specs = _by_name()
    assert specs["resolve"].required is True
    assert specs["assemble"].required is True


# ══ an optional stage must name its switch ═══════════════════════════
def test_every_optional_stage_names_its_toggle() -> None:
    for spec in DEFAULT_STAGES:
        if spec.required:
            continue
        assert spec.toggle, (
            f"{spec.name} is optional but names no config field, so nobody can tell how to turn it off"
        )


def test_the_toggle_table_matches_the_specs() -> None:
    expected = {spec.name: spec.toggle for spec in DEFAULT_STAGES if spec.toggle}
    assert dict(OPTIONAL_STAGE_TOGGLES) == expected, (
        "the toggle table and the stage specs disagree; one of them will go stale"
    )


# ══ one registry, not two lists ═══════════════════════════════════════
def test_the_registry_covers_every_stage() -> None:
    registry = stage_registry()
    assert set(registry) == {spec.name for spec in DEFAULT_STAGES}


def test_the_registry_entries_are_the_specs() -> None:
    registry = stage_registry()
    for spec in DEFAULT_STAGES:
        assert registry[spec.name] is spec


# ══ the order is still the order ═════════════════════════════════════
def test_the_declared_order_is_unchanged() -> None:
    assert [spec.name for spec in DEFAULT_STAGES] == [
        "resolve",
        "assemble",
        "claims",
        "cues",
        "temporal",
        "validate",
        "dedupe",
        "derive",
    ]


def test_required_outputs_are_relative_work_paths() -> None:
    for spec in DEFAULT_STAGES:
        for item in spec.required_outputs:
            assert "/" in item or item.endswith(".jsonl"), (
                f"{spec.name} names {item!r}, which is not a work-relative table path"
            )
