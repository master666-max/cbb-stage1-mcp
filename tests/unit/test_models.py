from cbb_stage1.models import EngineOutput


def test_engine_output_empty_is_valid():
    out = EngineOutput(unit_id="u:test")
    assert out.gleaning_round == 0
    assert out.abstain is False
