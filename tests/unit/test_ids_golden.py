from __future__ import annotations

from cbb_stage1 import ids


def test_contract_id_golden_vectors():
    """Pin the hash algorithm, not a particular book.

    The inputs are synthetic on purpose. The first version used a character name
    from the source novel, so the repository's copyright sanitisation changed the
    name and every name-derived id with it — a golden test going red because the
    input changed, which tells you nothing about the algorithm it exists to pin.
    """
    roles = {"主体": "e:zj:person:1", "值": {"literal": "A"}}
    quote_sha = "sha256:" + "1" * 64
    assert ids.assertion_id("zj", "性质", roles, quote_sha) == "a:828a784dff112e25"
    assert ids.entity_id("zj", "person", "测试人名") == "e:zj:person:f35ceee72d34"
    assert ids.event_id("zj", "相遇", "u:zj:v01:c001", ["e:2", "e:1"], "抵达") == "ev:0956c4fb85eb3d78"
    assert ids.cue_id("zj", "object", {"literal": "影子"}, "u:zj:v01:c001") == "cue:c20634a01d025192"
    assert ids.claim_id("a:" + "1" * 16, "性质：A") == "cl:44700357e9ff6147"
    assert ids.timeline_id(
        "zj", "before", "event", "ev:" + "1" * 16, "event", "ev:" + "2" * 16
    ) == "t:84581cc34dc1d546"
    assert ids.merge_id("zj", "alias", ["e:2", "e:1"], "e:3") == "mg:5aaaf9b2ea711aaa"
    assert ids.proposal_id("zj", "性质", ["u:2", "u:1"]) == "pp:1532f33160b09ee2"
    assert ids.mention_id("u:1", 1, 3, "示") == "m:u:1:1-3:f00a69ee"
    assert ids.unit_id("zj", 1, 2) == "u:zj:v01:c002"
    assert ids.summary_id("zj", "chapter", "u:1") == "sm:b03ce60d21273b96"
    assert ids.hypothesis_id("zj", "setting", "x") == "hy:f8371ea7d3b8dcba"
    assert ids.abduction_hypothesis_id("zj", "setting", "x", ["p1", "p2"]) == "hy:8b7f9acf2a8518c2"
    assert ids.community_id("zj", 1, ["e:2", "e:1"]) == "cm:ef44c4cf0186aee6"
    assert ids.report_id("cm:" + "1" * 16) == "cr:1707926a6a799ff9"
    assert ids.raptor_cluster_id("zj", 1, 1, ["u:2", "u:1"]) == "rt:deea3afd825f8ba9"
    assert ids.forecast_id("zj", "u:1") == "fc:635db6eb37add406"
    assert ids.conflict_id("zj", "attribute", "a1", "a2", "detail") == "c:4190861ac28c01a4"
    assert ids.quarantine_id({"record": {"x": 1}}) == "q:7a5c4c034e822a6f"
    assert ids.citation_id("u:1", 1, 3, "示") == "c:7e4ab58f0bb789e7"
    assert ids.sha256_text("abc") == (
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert ids.short_hash("abc") == "ba7816bf8f01cfea"
