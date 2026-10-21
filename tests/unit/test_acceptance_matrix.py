from pathlib import Path

from cbb_stage1.qc.acceptance import EXPECTED, load_matrix


def test_matrix_has_all_87_ids():
    matrix = Path(__file__).resolve().parents[2] / "acceptance" / "07.yaml"
    items = load_matrix(matrix)
    assert len(items) == 87
    assert [i["id"] for i in items] == EXPECTED


def test_matrix_statuses_are_known():
    matrix = Path(__file__).resolve().parents[2] / "acceptance" / "07.yaml"
    for item in load_matrix(matrix):
        assert item["expected"] in {"PASS", "PARTIAL", "EXCLUDED"}
