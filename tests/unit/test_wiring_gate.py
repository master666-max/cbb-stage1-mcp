from pathlib import Path

from cbb_stage1.qc.wiring import dead_modules


def test_no_functional_module_is_unreachable_from_entrypoints():
    assert dead_modules(Path(__file__).resolve().parents[2] / "src" / "cbb_stage1") == []
