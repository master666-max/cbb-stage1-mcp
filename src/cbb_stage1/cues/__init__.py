"""P3 cue package: the foreshadowing state machine (pure functions) + P0-C wiring.

Contract: box ``knowledge/cues.jsonl`` (design/01 §5.9/§5.18), invariants
I18/I19/I24. The state machine is code, never an LLM free-text judgement.
``cbb_stage1.cues.integration.run`` advances the candidates from the work layer
and writes ``work/assemble/cues.jsonl``.
"""

from __future__ import annotations

from .integration import run
from .state import (
    CUE_EVENTS,
    CUE_KINDS,
    CUE_STATES,
    CueTransition,
    CueViolation,
    apply_event,
    can_transition,
    cue_id_of,
    transition_cue,
    validate_cue,
)

__all__ = [
    "CUE_EVENTS", "CUE_KINDS", "CUE_STATES", "CueTransition", "CueViolation", "apply_event",
    "can_transition", "cue_id_of", "run", "transition_cue", "validate_cue",
]
