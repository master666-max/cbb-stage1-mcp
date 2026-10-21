from __future__ import annotations

import json
from pathlib import Path

from cbb_stage1.temporal.integration import run as temporal_run


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_temporal_solver_consumes_extracted_time_expressions(tmp_path):
    work = tmp_path / "work"
    text = "三日后，贝塔心叶抵达。次日，黑之魔王离开。"
    text_path = work / "plan" / "texts" / "u001.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    unit = {
        "unit_id": "u:t:v01:c001",
        "order": 1,
        "volume": 1,
        "text_path": "plan/texts/u001.txt",
    }
    (work / "plan").mkdir(parents=True, exist_ok=True)
    (work / "plan" / "units.jsonl").write_text(json.dumps(unit) + "\n", encoding="utf-8")
    events = [
        {
            "event_id": "ev:" + "1" * 16,
            "unit_id": unit["unit_id"],
            "type": "抵达",
            "summary": "三日后，贝塔心叶抵达",
            "evidence": [{"unit_id": unit["unit_id"], "start": 0, "end": 3, "quote": "三日后"}],
        },
        {
            "event_id": "ev:" + "2" * 16,
            "unit_id": unit["unit_id"],
            "type": "离开",
            "summary": "次日，黑之魔王离开",
            "evidence": [{"unit_id": unit["unit_id"], "start": 9, "end": 11, "quote": "次日"}],
        },
    ]
    _write(work / "assemble" / "events.jsonl", events)

    result = temporal_run(
        work,
        book_id="t",
        outputs=[],
        units=[unit],
        created_at="2026-10-19T00:00:00+08:00",
        upstream={},
    )
    assert result["counts"]["timeline_generated"] == 1
    assert result["counts"]["timeline_written"] == 1
    rows = [
        json.loads(line)
        for line in (work / "assemble" / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["kind"] == "before"
    assert rows[0]["left"]["id"] == events[0]["event_id"]
    assert rows[0]["right"]["id"] == events[1]["event_id"]
