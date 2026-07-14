from __future__ import annotations

from typing import Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, k: int = DEFAULT_RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)
    return sorted(scores, key=lambda item_id: scores[item_id], reverse=True)
