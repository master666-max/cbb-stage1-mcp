from cbb_stage1.box import projection_tables as pt


def _unit():
    return {"unit_id": "u:zj:v01:c001", "order": 1, "sha256": "sha256:" + "a" * 64}


def test_citations_are_content_addressed_and_exact():
    unit = _unit()
    ev = {"unit_id": unit["unit_id"], "start": 0, "end": 2, "quote": "甲乙", "occurrence": 1}
    citations = pt.build_citations(
        "zj", [unit], [{"evidence": [ev]}], [], [], [], []
    )
    assert len(citations) == 1
    c = citations[0]
    assert c["citation_id"].startswith("c:")
    assert c["source_hash"] == unit["sha256"]
    assert c["quote"] == "甲乙"


def test_narrative_anchor_and_coverage_60_cells():
    unit = _unit()
    text = "甲乙。\n「对话」\n"
    narratives = pt.build_narrative([unit], {unit["unit_id"]: text})
    pt.attach_narrative_anchors(narratives)
    citations = pt.build_citations("zj", [unit], [], [], [], [], narratives)
    assert len(narratives) == 2
    assert narratives[0]["anchor"] in {c["citation_id"] for c in citations}
    coverage = pt.build_coverage(
        "zj", [], [], [], [], [], [], [], citations, [], [], "2026-10-18T19:00:00+08:00"
    )
    assert len(coverage["cells"]) == 60
    assert all("status" in cell for cell in coverage["cells"])


def test_graph_edges_point_back_to_assertion():
    a = {
        "assertion_id": "a:" + "b" * 16,
        "predicate": "敌对",
        "library": "关系",
        "roles": {"施事": "e:zj:person:1", "受事": "e:zj:person:2"},
        "confidence": 0.8,
        "evidence": [],
        "status": "asserted",
    }
    edges = pt.build_graph_edges("zj", [a])
    assert len(edges) == 1
    assert edges[0]["assertion_id"] == a["assertion_id"]
    assert edges[0]["from"] == "e:zj:person:1"
    assert edges[0]["to"] == "e:zj:person:2"
