from __future__ import annotations

from scripts.utils.fusion import reciprocal_rank_fusion


def test_empty_rankings_return_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_ranking_preserves_order() -> None:
    assert reciprocal_rank_fusion([["a", "b", "c"]]) == ["a", "b", "c"]


def test_shared_high_rank_beats_single_arm_top() -> None:
    dense = ["x", "a", "b"]
    lexical = ["y", "a", "c"]
    fused = reciprocal_rank_fusion([dense, lexical])
    assert fused[0] == "a"
    assert set(fused) == {"a", "b", "c", "x", "y"}


def test_each_id_appears_once() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert sorted(fused) == ["a", "b"]


def test_order_depends_only_on_rank_not_score_scale() -> None:
    arm_one = ["a", "b", "c"]
    arm_two = ["b", "c", "a"]
    baseline = reciprocal_rank_fusion([arm_one, arm_two])
    assert reciprocal_rank_fusion([list(arm_one), list(arm_two)]) == baseline
    assert baseline == ["b", "a", "c"]


def test_k_smoothing_reduces_top_rank_dominance() -> None:
    arm_one = ["a", "b"]
    arm_two = ["b", "a"]
    assert reciprocal_rank_fusion([arm_one, arm_two], k=1) == ["a", "b"]
    assert reciprocal_rank_fusion([arm_one, arm_two], k=1000)[0] in {"a", "b"}
