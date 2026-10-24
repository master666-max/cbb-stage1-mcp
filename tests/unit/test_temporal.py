"""Unit tests for :mod:`cbb_stage1.temporal` (partial order + NLI interface).

Box contract: design/01 §5.8, invariants I5/I20/I25. The tests pin both the
positive path (partial order, no invented total order) and the negative paths
demanded by the Lead: **cycle -> quarantine**, contradiction, ref_missing, and
NLI disagreement that becomes a conflict instead of overwriting the constraint.
"""

from __future__ import annotations

import copy

import pytest

from cbb_stage1 import ids as contract_ids
from cbb_stage1.temporal import (
    Constraint, NliResult, TimelineContractError, find_cycles, make_conflict,
    nli_pairwise, normalize_result, parse_constraint, parse_constraints, question_for,
    solve, topological_order, verify_constraint,
)
from cbb_stage1.temporal.solver import Node


def rec(cid, kind, left, right, **kw):
    record = {"constraint_id": cid, "book_id": "zj", "kind": kind,
              "left": {"kind": "event", "id": left},
              "right": {"kind": "event", "id": right}}
    record.update(kw)
    return record


def chain():
    return [rec("t:aa", "before", "ev:a", "ev:b"),
            rec("t:bb", "before", "ev:b", "ev:c")]


def mock_nli(result, *, verifier="mock-nli", model="tiny-1"):
    calls = []

    def impl(left, right, question):
        calls.append((left, right, question))
        return {"result": result, "verifier": verifier, "model": model}

    impl.calls = calls
    return impl


# ---------------------------------------------------------------------------
# parsing / contract
# ---------------------------------------------------------------------------
def test_parse_constraint_minimal_and_stable_id():
    c1 = parse_constraint(rec(None, "before", "ev:a", "ev:b"))
    c2 = parse_constraint(rec(None, "before", "ev:a", "ev:b"))
    assert isinstance(c1, Constraint)
    assert c1.constraint_id == c2.constraint_id
    assert c1.constraint_id.startswith("t:")
    # 缺号用统一 t: 公式（cbb_stage1.ids.timeline_id）
    assert c1.constraint_id == contract_ids.timeline_id(
        "zj", "before", "event", "ev:a", "event", "ev:b")
    assert 16 <= len(c1.constraint_id) - 2 <= 64
    assert c1.left == Node("event", "ev:a")


def test_parse_constraint_defaults_and_as_record():
    c = parse_constraint(rec("t:aa", "after", "ev:a", "ev:b",
                             confidence=0.6, source="inferred", status="inferred",
                             evidence=[{"unit_id": "u:1", "quote": "三日后"}],
                             deps=["ev:a", "ev:b"]))
    out = c.as_record()
    assert c.constraint_id == "t:aa"      # 已有 id 原样保留，不重编号
    assert out["kind"] == "after" and out["source"] == "inferred"
    assert out["confidence"] == 0.6 and out["status"] == "inferred"
    assert out["evidence"][0]["quote"] == "三日后"
    assert out["deps"] == ["ev:a", "ev:b"]
    assert out["verify"] is None


@pytest.mark.parametrize("bad", [
    dict(kind="then"),
    dict(source="guessed"),
    dict(status="probably"),
    dict(confidence=1.5),
    dict(confidence="high"),
])
def test_parse_constraint_rejects_closed_set_violations(bad):
    record = rec("t:aa", "before", "ev:a", "ev:b")
    record.update(bad)
    with pytest.raises(TimelineContractError):
        parse_constraint(record)


def test_parse_constraint_rejects_bad_nodes():
    record = rec("t:aa", "before", "ev:a", "ev:b")
    record["left"] = {"kind": "chapter", "id": "ev:a"}
    with pytest.raises(TimelineContractError):
        parse_constraint(record)
    record = rec("t:aa", "before", "ev:a", "ev:b")
    record["right"] = {"kind": "event", "id": ""}
    with pytest.raises(TimelineContractError):
        parse_constraint(record)


def test_parse_constraint_rejects_bad_verify_enum():
    record = rec("t:aa", "before", "ev:a", "ev:b",
                 verify={"method": "llm-vibes", "result": "yes"})
    with pytest.raises(TimelineContractError):
        parse_constraint(record)
    record = rec("t:aa", "before", "ev:a", "ev:b",
                 verify={"method": "nli_pairwise", "result": "maybe"})
    with pytest.raises(TimelineContractError):
        parse_constraint(record)


def test_parse_constraints_splits_rejected_into_schema_quarantine():
    good, rejected = parse_constraints([rec("t:aa", "before", "ev:a", "ev:b"),
                                        dict(rec("t:bb", "then", "ev:b", "ev:c"))])
    assert len(good) == 1
    assert len(rejected) == 1
    assert rejected[0].reason == "schema"
    assert rejected[0].kind == "timeline"
    assert "闭集" in rejected[0].detail


# ---------------------------------------------------------------------------
# partial order
# ---------------------------------------------------------------------------
def test_before_chain_gives_partial_order_and_no_invented_total_order():
    result = solve(chain())
    assert result.has_cycle is False
    assert result.topological_order == ("ev:a", "ev:b", "ev:c")
    assert result.precedes("ev:a", "ev:b") is True
    assert result.precedes("ev:a", "ev:c") is True
    assert result.precedes("ev:b", "ev:a") is False
    assert result.relates("ev:a", "ev:c") == "before"
    assert result.relates("ev:c", "ev:a") == "after"
    assert result.relates("ev:b", "ev:b") == "same"


def test_unrelated_nodes_are_unknown_not_ordered():
    result = solve([rec("t:aa", "before", "ev:a", "ev:b")])
    assert result.relates("ev:a", "ev:z") == "unknown"
    assert result.relates("ev:z", "ev:a") == "unknown"
    # the solver did not invent an edge between incomparable nodes
    assert "ev:z" not in result.reachability.get("ev:a", frozenset())


def test_after_reverses_edge_direction():
    result = solve([rec("t:aa", "after", "ev:a", "ev:b")])
    assert result.relates("ev:b", "ev:a") == "before"
    assert result.relates("ev:a", "ev:b") == "after"


def test_isolated_node_is_included_in_order():
    result = solve([rec("t:aa", "before", "ev:a", "ev:b"),
                    {"constraint_id": "t:cc", "book_id": "zj", "kind": "overlap",
                     "left": {"kind": "event", "id": "ev:x"},
                     "right": {"kind": "event", "id": "ev:y"}}])
    assert set(result.nodes) == {"ev:a", "ev:b", "ev:x", "ev:y"}
    assert result.topological_order is not None
    assert result.relates("ev:x", "ev:y") == "unknown"


# ---------------------------------------------------------------------------
# cycles -> quarantine (I5)
# ---------------------------------------------------------------------------
def test_three_cycle_is_quarantined_and_no_order_is_emitted():
    records = chain() + [rec("t:cc", "before", "ev:c", "ev:a")]
    result = solve(records)
    assert result.has_cycle is True
    assert result.topological_order is None
    assert len(result.cycles) == 1
    reasons = [q.reason for q in result.quarantine]
    assert reasons and set(reasons) == {"timeline_cycle"}
    # quarantine ids use the unified q: formula; offending constraint ids stay
    # named in the detail (旧合法 t: id preserved, never renumbered)
    assert len(result.quarantine) == 3
    assert all(q.item_id.startswith("q:") for q in result.quarantine)
    for cid in ("t:aa", "t:bb", "t:cc"):
        assert any(cid in q.detail for q in result.quarantine)
    assert all(c.kind == "timeline" and c.status == "open" for c in result.conflicts)
    assert "回炉重抽" in result.quarantine[0].detail


def test_self_loop_is_a_cycle():
    result = solve([rec("t:self", "before", "ev:a", "ev:a")])
    assert result.has_cycle is True
    assert result.topological_order is None
    assert result.quarantine[0].reason == "timeline_cycle"


def test_before_and_after_same_pair_form_two_cycle():
    result = solve([rec("t:aa", "before", "ev:a", "ev:b"),
                    rec("t:bb", "after", "ev:a", "ev:b")])
    assert result.has_cycle is True
    assert set(result.cycles[0]) == {"ev:a", "ev:b"}
    assert len(result.quarantine) == 2


def test_cycle_does_not_stop_ordering_the_clean_subgraph():
    records = chain() + [rec("t:cc", "before", "ev:c", "ev:a"),
                         rec("t:dd", "before", "ev:d", "ev:e")]
    result = solve(records)
    assert result.has_cycle is True
    assert result.topological_order is None
    assert result.precedes("ev:d", "ev:e") is True
    assert result.quarantine and all(q.reason == "timeline_cycle" for q in result.quarantine)


def test_find_cycles_public_helper_dedupes_rotation():
    cycles = find_cycles({"a": {"b"}, "b": {"a"}})
    assert len(cycles) == 1
    assert set(cycles[0]) == {"a", "b"}
    assert find_cycles({"a": {"b"}, "b": {"c"}, "c": set()}) == ()


def test_topological_order_public_helper_returns_none_on_cycle():
    assert topological_order({"a", "b"}, {"a": {"b"}, "b": {"a"}}) is None
    assert topological_order({"a", "b"}, {"a": {"b"}}) == ("a", "b")


# ---------------------------------------------------------------------------
# during / overlap contradictions
# ---------------------------------------------------------------------------
def test_during_contradicts_strict_order():
    result = solve([rec("t:aa", "before", "ev:a", "ev:b"),
                    rec("t:dd", "during", "ev:a", "ev:b")])
    assert result.has_cycle is False
    item = next(q for q in result.quarantine if "t:dd" in q.detail)
    assert item.reason == "timeline_contradiction"
    assert any(c.left == "ev:a" and c.right == "ev:b" for c in result.conflicts)


def test_overlap_contradicts_strict_order():
    result = solve([rec("t:aa", "before", "ev:a", "ev:b"),
                    rec("t:oo", "overlap", "ev:a", "ev:b")])
    assert any("t:oo" in q.detail and q.reason == "timeline_contradiction"
               for q in result.quarantine)


def test_during_without_strict_order_is_kept():
    result = solve([rec("t:dd", "during", "ev:a", "ev:b")])
    assert not result.quarantine
    assert len(result.constraints) == 1


# ---------------------------------------------------------------------------
# reference closure
# ---------------------------------------------------------------------------
def test_unknown_event_reference_is_quarantined_not_solver_input():
    result = solve([rec("t:aa", "before", "ev:a", "ev:b")], known_events={"ev:a"})
    assert [q.reason for q in result.quarantine] == ["ref_missing"]
    assert result.constraints == ()
    assert result.topological_order == ()


# ---------------------------------------------------------------------------
# nli_pairwise interface (mockable; never network by default)
# ---------------------------------------------------------------------------
def test_nli_pairwise_default_impl_is_disabled_unknown():
    result = nli_pairwise("ev:a", "ev:b", question_for("before", "ev:a", "ev:b"))
    assert result.result == "unknown"
    assert result.verifier == "disabled"
    assert result.method == "nli_pairwise"


def test_verify_constraint_flags_contradiction():
    constraint = parse_constraint(rec("t:aa", "before", "ev:a", "ev:b"))
    record, contradicted = verify_constraint(constraint, mock_nli("no"))
    assert contradicted is True
    # The question now states that no event text was supplied. The old expectation was the id-only
    # prompt, which is the defect D11 fixed: a model cannot order two opaque ids, so an answer to
    # that question is a guess recorded as a verdict.
    assert record == {"method": "nli_pairwise",
                      "question": "ev:a （未提供事件正文，无法判断） 是否先于 ev:b （未提供事件正文，无法判断）？",
                      "result": "no", "verifier": "mock-nli", "model": "tiny-1"}
    record_yes, contradicted_yes = verify_constraint(constraint, mock_nli("yes"))
    assert contradicted_yes is False and record_yes["result"] == "yes"


def test_unknown_nli_result_is_unverified_not_passed():
    result = solve([rec("t:aa", "before", "ev:a", "ev:b")],
                   nli=lambda l, r, q: "unknown", verify=True)
    assert result.conflicts == ()
    # the constraint is kept with an explicit unknown verify record (never silently "passed")
    assert result.constraints[0].verify["result"] == "unknown"


def test_nli_contradiction_does_not_overwrite_constraint():
    result = solve([rec("t:aa", "before", "ev:a", "ev:b")], nli=mock_nli("no"), verify=True)
    assert len(result.constraints) == 1
    assert result.constraints[0].kind == "before"
    assert result.constraints[0].verify["result"] == "no"
    assert len(result.conflicts) == 1
    assert result.conflicts[0].kind == "timeline"
    assert "NLI 判 no" in result.conflicts[0].detail


def test_verify_only_selected_constraints():
    impl = mock_nli("yes")
    result = solve(chain(), nli=impl, verify=["t:aa"])
    verify_by_id = {c.constraint_id: c.verify for c in result.constraints}
    assert verify_by_id["t:aa"]["result"] == "yes"
    assert verify_by_id["t:bb"] is None
    assert len(impl.calls) == 1


def test_nli_result_validation_and_normalization():
    with pytest.raises(ValueError):
        NliResult(question="q", result="probably")
    with pytest.raises(ValueError):
        NliResult(question="q", method="vibes", result="yes")
    raw = normalize_result("no", question="q", verifier="v")
    assert raw.result == "no" and raw.verifier == "v"


def test_question_for_all_kinds_and_unknown():
    for kind in ("before", "after", "during", "overlap"):
        text = question_for(kind, "ev:a", "ev:b")
        assert "ev:a" in text and "ev:b" in text
    with pytest.raises(ValueError):
        question_for("then", "a", "b")


# ---------------------------------------------------------------------------
# purity + id stability
# ---------------------------------------------------------------------------
def test_solve_is_pure():
    records = chain()
    snapshot = copy.deepcopy(records)
    solve(records)
    assert records == snapshot


def test_conflict_ids_are_stable_and_box_shaped():
    one = make_conflict("zj", "ev:a", "ev:b", "环")
    two = make_conflict("zj", "ev:a", "ev:b", "环")
    assert one.conflict_id == two.conflict_id
    record = one.as_record()
    assert record["conflict_id"].startswith("c:")
    assert record["kind"] == "timeline" and record["status"] == "open"
    assert record["decision"] is None and record["resolved_by"] is None


def test_quarantine_record_shape():
    result = solve([rec("t:aa", "before", "ev:a", "ev:a")])
    record = result.quarantine[0].as_record()
    assert set(record) == {"item_id", "kind", "reason", "detail", "source_run"}
    assert record["kind"] == "timeline"
