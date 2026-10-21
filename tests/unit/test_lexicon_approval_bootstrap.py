"""An approved predicate must survive the rebuild.

Measured on a real run: ``cbb_lexicon_approve`` refused with

    lexicon not found at <work>/lexicon/predicates.json; pass --lexicon. The registry
    is the single source of truth, so it must exist before a predicate can be added to it.

and that refusal is correct in isolation — but the box build does not read a
registry the work tree never had. ``box/writer.py`` looks for ``<work>/lexicon/`` and
falls back to ``default_lexicon()`` when the directory is absent, so an approval made
against some other file would be silently dropped on the next build.

So the approval path materialises the registry from the project's own default when the
work tree has none, and the approved predicate then lands in the registry the box
actually reads. Nothing is invented: the default is this project's, and the materialised
files are byte-equivalent to what the build would have used anyway.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.assemble.lexicon_evolution import approve_proposal
from cbb_stage1.box.lexicon import load_lexicon

PROPOSAL = {
    "proposal_id": "pp:1",
    "book_id": "t",
    "predicate": "世界观声明",
    "seen_in": ["u:t:v01:c001"],
    "count": 1,
    "role_shape_examples": [{"类型": "layer_exists"}],
    "status": "proposed",
    "decided_by": None,
}


def _work(root: Path) -> Path:
    work = root / "work"
    path = work / "assemble" / "proposed_predicates.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(PROPOSAL, ensure_ascii=False) + "\n", encoding="utf-8")
    return work


def test_the_approval_works_on_a_work_tree_without_a_lexicon(tmp_path: Path) -> None:
    """The common case: the run never had a registry, and the box used the default."""
    work = _work(tmp_path)
    assert not (work / "lexicon").exists(), "precondition: no lexicon yet"

    result = approve_proposal(
        work, "世界观声明", approved_by="张三", library="设定"
    )

    assert result.predicate == "世界观声明"
    assert result.approved_by == "张三"
    assert (work / "lexicon" / "predicates.json").exists()


def test_the_materialised_registry_is_loadable_by_the_box_writer(tmp_path: Path) -> None:
    """If `load_lexicon` cannot read it, the box build would not use it either."""
    work = _work(tmp_path)
    approve_proposal(work, "世界观声明", approved_by="张三", library="设定")

    lexicon = load_lexicon(work / "lexicon")

    assert lexicon.genre_id
    assert lexicon.types, "types must survive the round trip"
    assert "世界观声明" in lexicon.predicates, "the approved predicate must be in it"
    assert "位于" in lexicon.predicates, "the default predicates must survive too"


def test_the_approved_predicate_keeps_its_library_and_approver(tmp_path: Path) -> None:
    work = _work(tmp_path)
    approve_proposal(work, "世界观声明", approved_by="张三", library="设定")

    lexicon = load_lexicon(work / "lexicon")
    spec = lexicon.predicates["世界观声明"]
    assert spec.library == "设定"

    payload = json.loads((work / "lexicon" / "predicates.json").read_text(encoding="utf-8"))
    approvers = [row.get("approved_by") for row in payload.get("approved") or ()]
    assert "张三" in approvers, "I4: the human gate must be recorded in the registry"


def test_an_existing_registry_is_not_reset_to_the_default(tmp_path: Path) -> None:
    """Materialising must never clobber a registry somebody curated.

    The registry is the single source of truth, so an approval must ADD to it. The
    failure mode to guard against is a bootstrap that overwrites a hand-made
    registry with the default — the curated predicates would vanish.
    """
    work = _work(tmp_path)
    lexicon_dir = work / "lexicon"
    lexicon_dir.mkdir(parents=True)
    (lexicon_dir / "predicates.json").write_text(
        json.dumps(
            {
                "contract": "cbb-lexicon/1",
                "predicates": [
                    {
                        "pred": "位于",
                        "library": "关系",
                        "roles": [
                            {"name": "主体", "type": "entity", "required": True},
                            {"name": "值", "type": "literal", "required": True},
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (lexicon_dir / "types.json").write_text(
        json.dumps({"types": {"实体": {"kind": "entity"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (lexicon_dir / "genre.json").write_text(
        json.dumps({"id": "light-novel-zh"}, ensure_ascii=False), encoding="utf-8"
    )

    approve_proposal(work, "世界观声明", approved_by="张三", library="设定")

    payload = json.loads((lexicon_dir / "predicates.json").read_text(encoding="utf-8"))
    names = {row["pred"] for row in payload["predicates"]}
    assert "位于" in names, "curated predicates must survive the approval"
    assert "世界观声明" in names, "the approved predicate must be added"


def test_the_proposal_row_is_marked_approved(tmp_path: Path) -> None:
    work = _work(tmp_path)
    approve_proposal(work, "世界观声明", approved_by="张三", library="设定")

    rows = [
        json.loads(line)
        for line in (work / "assemble" / "proposed_predicates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert rows[0]["status"] == "approved"
    assert rows[0]["decided_by"] == "张三"