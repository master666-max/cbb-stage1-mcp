"""Work/box path layout. Work is disposable; box is immutable."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkPaths:
    root: Path

    @property
    def plan(self) -> Path:
        return self.root / "plan"

    @property
    def engine(self) -> Path:
        return self.root / "engine"

    @property
    def verify(self) -> Path:
        return self.root / "verify"

    @property
    def resolve(self) -> Path:
        return self.root / "resolve"

    @property
    def assemble(self) -> Path:
        return self.root / "assemble"

    @property
    def claims(self) -> Path:
        return self.root / "claims"

    @property
    def box(self) -> Path:
        return self.root / "box"

    def ensure(self) -> "WorkPaths":
        for p in [self.plan, self.engine, self.verify, self.resolve, self.assemble, self.claims, self.box]:
            p.mkdir(parents=True, exist_ok=True)
        return self
