from pathlib import Path

from cbb_stage1.qc.id_sources import scan


def test_no_contract_id_formula_outside_ids_module():
    assert scan(Path(__file__).resolve().parents[2] / "src" / "cbb_stage1") == []
