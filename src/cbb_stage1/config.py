"""Runtime configuration for cbb_stage1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    book_id: str
    book_path: Path
    out_dir: Path
    work_dir: Path
    genre: str = "light-novel-zh"
    model: str = "mock"
    provider: str = "mock"
    max_gleanings: int = 1
    canonical_max_gleanings: int = 2
    minor_max_gleanings: int = 0
    chunk_target_tokens: int = 1200
    chunk_overlap_tokens: int = 100
    chunk_strategy: str = "structural"
    max_workers: int = 4
    live: bool = False
    #: D2: is the narrator of this book trustworthy?
    #:
    #: ``unreliable`` downgrades narration-only assertions to ``claim_layer=
    #: statement`` (01-箱规格 §5.16): if the narrator lies, their plain narration is
    #: a claim, not a fact. The plumbing already existed end to end — units carry
    #: the slot, the pipeline passes it, the assembler honours it — but nothing ever
    #: set it to anything but ``unknown``, so the mechanism could not fire.
    narrator_reliability: str = "unknown"
    #: B4 judge / D4: an OpenAI-compatible endpoint for the judging model.
    #:
    #: Must be a DIFFERENT model from the extractor; §5.15 requires the judgement
    #: not to come from the model that produced the claim. Empty means "no judge",
    #: which keeps ``model_decoupled=false`` honest rather than papering over it.
    judge_base_url: str = ""
    judge_model: str = ""
    #: B12 abstain gate: below this self-reported confidence a claim abstains instead of being
    #: promoted. Equal to `claims.verdicts.DEFAULT_ABSTAIN_THRESHOLD`; the two are pinned together
    #: by a test, because a knob whose default silently disagrees with the code's is worse than no
    #: knob. Nothing in the pipeline supplied this before, so the gate always used 0.5.
    abstain_threshold: float = 0.5
    nli_provider: str = "disabled"
    nli_model: str | None = None
    incremental: bool = False
    uq_head_checkpoint: str | None = None
    description_synthesis: bool = True
    summary_synthesis: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw["book_path"] = Path(raw["book_path"])
        raw["out_dir"] = Path(raw.get("out_dir", "artifacts/box"))
        raw["work_dir"] = Path(raw.get("work_dir", "artifacts/work"))
        return cls(**raw)

    def for_tier(self, tier: str) -> int:
        if tier == "canonical":
            return self.canonical_max_gleanings
        if tier == "minor":
            return self.minor_max_gleanings
        return self.max_gleanings
