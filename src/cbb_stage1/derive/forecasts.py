"""Chapter-level forecast proxy (100-Endings-style, theory only).

I deleted this file once, calling it a "fake implementation". That was wrong, and
the correction matters because it is the same mistake this project keeps making:
judging an artefact against an imagined standard instead of against its contract.

What ``01-箱规格`` §5.20 actually asks for::

    derived/inference/forecasts.jsonl（100-Endings 类张力/预测；论文只验证短篇，
    长篇未验证，只按章做辅助，且标"理论"）

That is: chapter-scoped, marked theoretical, never promoted to a fact. The module
does exactly that — ``scope="chapter"``, ``theory=True``, and it lives only under
``derived/inference/``. The spec never asked for the real 100-Endings method
(sentences × 100 generations ≈ 180,000 calls for this book); it asked for a
chapter-level auxiliary reading, which is what this is.

The honest label is "proxy", not "fake". The number is a lexical singleton ratio —
it measures lexical dispersion, NOT narrative tension, and the docstring said so
before I removed it. So the fix is to keep the code and make the limit impossible
to miss:

* ``method`` states what is actually computed;
* ``no_rate`` is named for what it is (a proxy), and ``theory`` stays true;
* MCP records the capability as DISABLED so nobody mistakes it for forecasting.

Restored from 6dc229a1.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[\u4e00-\u9fff]{1,4}|[A-Za-z]{2,}")

#: What this actually computes. Not narrative tension — see the module docstring.
METHOD = "lexical_singleton_ratio"

#: Why the real method is not implemented, and why this proxy exists instead.
LIMIT = (
    "代理指标，不是叙事张力：100-Endings 原法需要「句数 × 100 次生成」"
    "（本书 18 章约 18 万次调用）。规范 §5.20 只要求按章做理论辅助，"
    "本模块提供的是词法离散度，仅供编辑参考，不进事实卡。"
)


@dataclass(frozen=True)
class ChapterForecast:
    scope: str = "chapter"
    theory: bool = True
    no_rate: float = 1.0
    top_terms: list[str] = field(default_factory=list)
    note: str = "100-Endings 只验证短篇；本读数按章、仅作理论辅助"
    #: Which proxy produced ``no_rate``. Recorded so a reader cannot mistake it for
    #: the paper's metric.
    method: str = METHOD

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "theory": self.theory,
            "no_rate": self.no_rate,
            "top_terms": self.top_terms,
            "note": self.note,
            "method": self.method,
        }


def forecast_chapter(text: str, top_k: int = 8) -> ChapterForecast:
    """Deterministic chapter-level lexical-dispersion proxy; no model, no fact promotion.

    ``no_rate`` is the share of tokens that occur exactly once in the chapter. It is
    a proxy for lexical dispersion. It is NOT the paper's no-rate (fraction of 100
    sampled endings that diverge from the actual one), and must not be read as one.
    """
    tokens = _TOKEN.findall(text or "")
    if not tokens:
        return ChapterForecast(no_rate=1.0, top_terms=[])
    counts = Counter(tokens)
    top = [term for term, _ in counts.most_common(top_k)]
    singleton = sum(1 for n in counts.values() if n == 1)
    no_rate = round(singleton / max(1, len(counts)), 4)
    return ChapterForecast(no_rate=no_rate, top_terms=top)
