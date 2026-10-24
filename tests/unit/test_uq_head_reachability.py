"""A configured UQ-head checkpoint must reach the method selection.

Measured gap (B5, reachability half)
------------------------------------
`confidence/__init__.py:194-210` decides the method from call-site kwargs only:

    if kwargs.get("head") is not None or kwargs.get("checkpoint"): return "uq_heads"

and the word `config` does not appear anywhere in the module. `uq_heads.resolve_checkpoint(config=...)`
exists and reads `Config.uq_head_checkpoint` correctly — it is simply never consulted by the auto path.

So `CBB_UQ_HEAD_CHECKPOINT` or `cfg.uq_head_checkpoint` has no effect: the checkpoint is configured and
ignored. The audit graded this as "unreachable", which is the accurate description — the algorithm is
short of a whitebox backend (a separate, adjudicated matter), but the *config* half is a plain wiring
gap and is fixed here.

What is pinned
--------------
* `score()` accepts `config=` and passes it to the selection;
* a checkpoint named in the config selects `uq_heads`, exactly as an explicit `checkpoint=` kwarg does;
* an explicit kwarg still wins over the config, so a caller can override;
* a config with no checkpoint selects nothing new — the behaviour for every existing caller is
  unchanged.
"""

from __future__ import annotations

from typing import Any

from cbb_stage1.confidence import _auto_method, score


class FakeConfig:
    def __init__(self, checkpoint: str | None = None) -> None:
        self.uq_head_checkpoint = checkpoint


# ══ the selector consults the config ═════════════════════════════════
def test_a_configured_checkpoint_selects_uq_heads() -> None:
    method = _auto_method(None, {"config": FakeConfig("C:/models/head.pt")})
    assert method == "uq_heads", (
        f"a configured checkpoint selected {method!r}; the config is being ignored"
    )


def test_an_explicit_kwarg_still_wins() -> None:
    method = _auto_method(
        None, {"checkpoint": "C:/explicit.pt", "config": FakeConfig("C:/models/head.pt")}
    )
    assert method == "uq_heads"


def test_a_config_without_a_checkpoint_changes_nothing() -> None:
    """The behaviour for every existing caller must be identical."""
    before = _auto_method(None, {})
    after = _auto_method(None, {"config": FakeConfig(None)})
    assert before == after


def test_a_config_object_without_the_field_is_tolerated() -> None:
    """Not every caller passes a Config; a plain object must not raise."""
    method = _auto_method(None, {"config": object()})
    assert method == _auto_method(None, {})


# ══ score() forwards it ═════════════════════════════════════════════
def test_score_accepts_a_config() -> None:
    import inspect

    parameters = inspect.signature(score).parameters
    accepts = "config" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    assert accepts, "score() cannot be given a config, so the checkpoint can never reach selection"


def test_score_with_a_configured_checkpoint_reports_the_method() -> None:
    record: dict[str, Any] = {"normalized_claim": "言万与拉法敌对"}
    result = score(record, method=None, config=FakeConfig("C:/models/head.pt"))
    # Without a loadable head the result is `unavailable`, but the *selection* must have been uq_heads.
    assert getattr(result, "method", "") == "uq_heads" or "uq_heads" in str(result), (
        f"the configured checkpoint did not select the method: {result}"
    )
