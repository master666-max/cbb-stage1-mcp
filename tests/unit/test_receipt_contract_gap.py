"""The receipt contract is a dead file; make the gap visible instead of pretending it is met.

Measured (C2)
-------------
`schemas/exec/receipt.schema.json` requires 14 fields:

    task_id role status unit_id output_path output_sha256 counts new_entities anomalies
    tokens cost_usd started_at ended_at error

while `orchestrator/roles.py`'s `Receipt` carries five:

    task_id status outputs counts notes

and nothing anywhere references the schema — a repo-wide search for `schemas/exec` returns zero
hits. So the file asserts a contract that no code reads and no artefact satisfies.

Why it cannot simply be "fixed"
-------------------------------
Two of the missing fields cannot be known here. `tokens` and `cost_usd` are only available if the
model reports usage, and under `host` transport it does not (the doctor says so:
`token_accounting: only if the agent reports usage`). Writing `{"in": 0, "out": 0}` would state
"zero tokens were used", which is false in the same way a fabricated confidence is false — the
audit's own rule against inventing numbers applies to accounting too.

So the honest move is to make the shortfall **explicit and self-maintaining**: compute it from the
schema and the dataclass at runtime, so adding fields shrinks it and growing the schema surfaces it.
Then a reader (or a gate) can see exactly what is unmet instead of trusting a stale file.
"""

from __future__ import annotations

from cbb_stage1.orchestrator.roles import Receipt, receipt_contract_gaps

#: Fields the exec contract demands and the receipt does not carry, as of this audit.
KNOWN_MISSING = {
    "role",
    "unit_id",
    "output_path",
    "output_sha256",
    "new_entities",
    "anomalies",
    "tokens",
    "cost_usd",
    "started_at",
    "ended_at",
    "error",
}


# ══ the gap is computed, not asserted by hand ═════════════════════════
def test_the_gap_is_computed_from_the_schema() -> None:
    gaps = receipt_contract_gaps()
    assert gaps, "the receipt is reported as conformant; it carries 5 of 14 required fields"


def test_the_computed_gap_matches_the_known_one() -> None:
    assert set(receipt_contract_gaps()) == KNOWN_MISSING, (
        "the gap changed; update the declaration rather than letting it drift"
    )


def test_the_gap_is_sorted_and_unique() -> None:
    gaps = receipt_contract_gaps()
    assert list(gaps) == sorted(set(gaps))


# ══ what the receipt does carry must still be honest ══════════════════
def test_the_receipt_carries_the_five_fields_it_claims() -> None:
    assert set(Receipt.__dataclass_fields__) == {
        "task_id",
        "status",
        "outputs",
        "counts",
        "notes",
    }


def test_the_gap_function_is_documented_as_a_known_shortfall() -> None:
    doc = (receipt_contract_gaps.__doc__ or "").lower()
    assert "schema" in doc
    assert "not" in doc or "cannot" in doc, (
        "the function must say the shortfall is unresolved, not just list fields"
    )


def test_the_receipt_module_states_the_shortfall() -> None:
    """A reader of roles.py must not conclude the receipt satisfies the exec contract."""
    import inspect

    from cbb_stage1.orchestrator import roles

    source = inspect.getsource(roles).lower()
    assert "exec/receipt.schema.json" in source, (
        "roles.py does not mention the contract it does not satisfy"
    )
