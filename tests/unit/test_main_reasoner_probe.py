"""The capability set is not a project constant — it depends on the driver.

Under the agent-driven architecture the main reasoning is served by whichever
sub-agent is driving the session, and that can be a different model on every run.
That is the trade the architecture makes: free model switching, at the price of
not knowing up-front what the model can do.

So this cannot be a hard-coded assumption anywhere. It has to be *declared at
start-up*, which is what the ``main_reasoner`` probe does: it reports what the
transport itself determines, marks the rest as driver-dependent, and says which
mechanisms are affected.

The failure this prevents: B3 SEPs and B4 whitebox report ``unavailable`` three
stages into a run, and the operator has to work backwards to discover that the
driving model never exposed logprobs.
"""

from __future__ import annotations

from cbb_stage1.preflight import OBSERVED, _probe_main_reasoner


def test_host_transport_marks_capabilities_as_driver_dependent() -> None:
    result = _probe_main_reasoner(transport="host", session_model_label="deepseek-v4.1")
    assert result["status"] == OBSERVED
    caps = result["evidence"]["capabilities"]
    # The transport itself cannot determine these — only the driver can.
    assert caps["token_logprobs"] == "unknown"
    assert caps["hidden_states"].startswith("no")
    assert caps["stable_model_identity"].startswith("no")
    assert caps["second_model"].startswith("no")
    assert "deepseek-v4.1" in result["detail"]


def test_host_transport_without_a_label_says_so() -> None:
    """Without a label the box cannot record who produced it."""
    result = _probe_main_reasoner(transport="host")
    assert "unlabelled" in result["detail"]
    assert result["evidence"]["session_model_label"] is None


def test_http_transport_determines_more_than_host() -> None:
    result = _probe_main_reasoner(
        transport="http", session_model_label="gpt-4o", base_url="https://api.openai.com/v1"
    )
    caps = result["evidence"]["capabilities"]
    assert caps["stable_model_identity"] == "yes"
    assert caps["token_accounting"] == "yes"
    assert caps["token_logprobs"] == "depends on vendor"


def test_a_vendor_without_logprobs_is_reported_as_excluded() -> None:
    """Anthropic does not expose token logprobs; that is knowable from the URL."""
    result = _probe_main_reasoner(
        transport="http",
        session_model_label="claude-sonnet",
        base_url="https://api.anthropic.com/v1",
    )
    caps = result["evidence"]["capabilities"]
    assert "vendor does not expose" in caps["token_logprobs"]
    assert result["evidence"]["logprob_verdict"] == "vendor-excluded"


def test_mcp_transport_defers_to_the_backend() -> None:
    result = _probe_main_reasoner(transport="mcp", session_model_label="local-llama")
    caps = result["evidence"]["capabilities"]
    assert caps["token_logprobs"] == "depends on backend"
    assert caps["stable_model_identity"] == "yes"


def test_probe_never_blocks() -> None:
    """An unknown capability set is normal here, not a defect.

    It only becomes a blocker when a mechanism that needs a signal is enabled
    while the signal is absent — which the per-capability probes report.
    """
    for transport in ("host", "mcp", "http"):
        result = _probe_main_reasoner(transport=transport)
        assert result["status"] == OBSERVED, transport


def test_probe_is_registered_before_the_capability_probes() -> None:
    """Cause should be listed before the symptoms it explains."""
    from cbb_stage1.preflight import PROBES

    names = [name for name, _ in PROBES]
    assert "main_reasoner" in names
    assert names.index("main_reasoner") < names.index("judge_backend")
    assert names.index("main_reasoner") < names.index("whitebox_backend")
