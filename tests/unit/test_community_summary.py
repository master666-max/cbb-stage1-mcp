from __future__ import annotations

from cbb_stage1.derive.community import build_community_reports


def test_community_summary_receives_assertion_evidence_and_source_ids():
    prompts = []

    def summarizer(prompt: str) -> str:
        prompts.append(prompt)
        return '{"title":"旧友","summary":"两人在船上相遇。"}'

    rows = build_community_reports(
        "b",
        [
            {
                "assertion_id": "a:1",
                "predicate": "相遇",
                "roles": {"施事": "e:1", "受事": "e:2"},
                "evidence": [{"unit_id": "u:1", "quote": "两人在船上相遇"}],
            }
        ],
        summarizer=summarizer,
    )
    assert rows
    assert "两人在船上相遇" in prompts[0]
    assert rows[0]["title"] == "旧友"
    assert rows[0]["summary"] == "两人在船上相遇。"
    assert rows[0]["source_assertion_ids"] == ["a:1"]
