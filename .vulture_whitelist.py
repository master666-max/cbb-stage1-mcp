"""Vulture whitelist: framework magic and deliberate public API only.

Vulture cannot see what a framework calls. Pydantic instantiates validators and
reads ``model_config`` at class-creation time; nothing in the source text calls
them. Those names are listed here so the remaining output is signal.

Rule for adding a line: it must be either (a) invoked by a framework, or (b) part
of the public contract that another package imports. "I might need it later" is
not a reason — that is how the 52-function backlog happened.
"""

# ── Pydantic: validators and config are invoked by the metaclass ──────────────
from cbb_stage1.assemble.registry import PredicateRegistry, PredicateSpec, RoleSpec

PredicateSpec.model_config
RoleSpec.model_config
PredicateRegistry.model_config

# ── Pydantic: field/model validators ─────────────────────────────────────────
from cbb_stage1.assemble.assertions import Assertion, QuarantineItem

QuarantineItem._validate_reason

# ── dataclass/Protocol surface read by reflection ────────────────────────────
from cbb_stage1.claims.verdicts import PromotionDecision

PromotionDecision.independent_source_count

# ── context manager protocol: __exit__ receives exc_tb from the interpreter ──
from cbb_stage1.orchestrator.checkpoint import CheckpointManager
from cbb_stage1.orchestrator.tracer import TracedSpan

CheckpointManager.__exit__
TracedSpan.__exit__
