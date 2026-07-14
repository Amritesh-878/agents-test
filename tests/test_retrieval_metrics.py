from __future__ import annotations

from scripts.utils.retrieval_metrics import (
    HIGH_SCORE_THRESHOLD,
    MEDIUM_SCORE_THRESHOLD,
    aggregate_metrics,
    classify_case_status,
    compute_case_metrics,
    first_hit_rank,
    is_hit_at_k,
    normalize_for_containment,
    provisional_grade,
    provisional_top_and_gap,
    quote_coverage,
    quote_in_text,
)

RETRIEVED = [
    "The intercept is it should be negative in the supply function.",
    "taxes, then technology, price of related goods, complementary goods, determinant",
    "unrelated chatter about the weather",
]


def test_normalize_collapses_whitespace_and_case() -> None:
    assert normalize_for_containment("  Hello   WORLD  ") == "hello world"


def test_normalize_strips_spaced_punctuation() -> None:
    quote = "Nisha , do we have to send our answers ?"
    text = "So Nisha, do we have to send our answers? she asked."
    assert normalize_for_containment(quote) in normalize_for_containment(text)


def test_normalize_bridges_slash_spacing_variants() -> None:
    assert normalize_for_containment("20/11") == normalize_for_containment("20 / 11")


def test_normalize_preserves_devanagari() -> None:
    assert normalize_for_containment("नमस्ते, hello") == "नमस्ते hello"


def test_quote_in_text_positive_and_negative() -> None:
    assert quote_in_text("intercept is it should be negative", RETRIEVED[0])
    assert not quote_in_text("balance of payments", RETRIEVED[0])


def test_quote_in_text_empty_quote_is_not_a_hit() -> None:
    assert not quote_in_text("   ", RETRIEVED[0])


def test_first_hit_rank_is_one_indexed() -> None:
    assert first_hit_rank(RETRIEVED, ["complementary goods"]) == 2


def test_first_hit_rank_none_when_absent() -> None:
    assert first_hit_rank(RETRIEVED, ["gross domestic product"]) is None


def test_first_hit_rank_matches_on_any_quote() -> None:
    assert first_hit_rank(RETRIEVED, ["not present", "price of related goods"]) == 2


def test_classify_refusal_when_flagged_or_no_quotes() -> None:
    assert classify_case_status(["x"], ["x"], is_refusal=True) == "refusal"
    assert classify_case_status([], ["x"], is_refusal=False) == "refusal"


def test_classify_scorable_when_quote_indexed() -> None:
    assert classify_case_status(["complementary goods"], RETRIEVED, is_refusal=False) == "scorable"


def test_classify_quote_not_indexed_when_absent_from_universe() -> None:
    status = classify_case_status(["gross domestic product"], RETRIEVED, is_refusal=False)
    assert status == "quote_not_indexed"


def test_is_hit_at_k_respects_cutoff_and_status() -> None:
    assert is_hit_at_k("scorable", 5, 5)
    assert not is_hit_at_k("scorable", 6, 5)
    assert not is_hit_at_k("scorable", None, 5)
    assert not is_hit_at_k("quote_not_indexed", 1, 5)
    assert not is_hit_at_k("refusal", 1, 5)


def test_top_and_gap_handles_zero_one_many() -> None:
    assert provisional_top_and_gap([]) == (None, None)
    assert provisional_top_and_gap([0.7]) == (0.7, 0.7)
    top, gap = provisional_top_and_gap([0.3, 0.9, 0.6])
    assert top == 0.9
    assert gap == round(0.9 - 0.6, 6)


def test_top_and_gap_ignores_none_scores() -> None:
    assert provisional_top_and_gap([None, 0.5, None]) == (0.5, 0.5)


def test_grade_high_needs_score_and_gap() -> None:
    assert provisional_grade([HIGH_SCORE_THRESHOLD, 0.5]) == "high"
    assert provisional_grade([0.9]) == "high"


def test_grade_medium_when_gap_too_small_but_score_ok() -> None:
    assert provisional_grade([0.8, 0.79]) == "medium"


def test_grade_medium_when_score_below_high() -> None:
    assert provisional_grade([MEDIUM_SCORE_THRESHOLD, 0.1]) == "medium"


def test_grade_low_when_weak_or_empty() -> None:
    assert provisional_grade([0.44]) == "low"
    assert provisional_grade([]) == "low"


def test_compute_case_metrics_scorable_hit() -> None:
    metric = compute_case_metrics(
        quotes=["complementary goods"],
        retrieved_texts=RETRIEVED,
        indexed_texts=RETRIEVED,
        scores=[0.8, 0.6, 0.4],
        is_refusal=False,
        k=5,
    )
    assert metric.status == "scorable"
    assert metric.first_hit_rank == 2
    assert metric.hit
    assert metric.tier == "high"
    assert [coverage.found_in_retrieved for coverage in metric.quote_coverage] == [True]


def test_compute_case_metrics_retrieved_beyond_k_is_not_a_hit() -> None:
    metric = compute_case_metrics(
        quotes=["price of related goods"],
        retrieved_texts=RETRIEVED,
        indexed_texts=RETRIEVED,
        scores=[0.8, 0.6, 0.4],
        is_refusal=False,
        k=1,
    )
    assert metric.first_hit_rank == 2
    assert not metric.hit


def test_compute_case_metrics_quote_not_indexed() -> None:
    metric = compute_case_metrics(
        quotes=["gross domestic product"],
        retrieved_texts=RETRIEVED,
        indexed_texts=RETRIEVED,
        scores=[0.8, 0.6],
        is_refusal=False,
        k=5,
    )
    assert metric.status == "quote_not_indexed"
    assert not metric.hit
    assert metric.first_hit_rank is None


def test_aggregate_excludes_refusal_and_not_indexed_from_recall() -> None:
    scorable_hit = compute_case_metrics(
        quotes=["intercept is it should be negative"],
        retrieved_texts=RETRIEVED,
        indexed_texts=RETRIEVED,
        scores=[0.9, 0.5],
        is_refusal=False,
        k=5,
    )
    scorable_miss = compute_case_metrics(
        quotes=["quantity supply is negative 12"],
        retrieved_texts=RETRIEVED,
        indexed_texts=[*RETRIEVED, "quantity supply is negative 12 when the price is one"],
        scores=[0.2, 0.1],
        is_refusal=False,
        k=5,
    )
    refusal = compute_case_metrics(
        quotes=[],
        retrieved_texts=RETRIEVED,
        indexed_texts=RETRIEVED,
        scores=[0.2],
        is_refusal=True,
        k=5,
    )
    not_indexed = compute_case_metrics(
        quotes=["gross domestic product"],
        retrieved_texts=RETRIEVED,
        indexed_texts=RETRIEVED,
        scores=[0.8, 0.6],
        is_refusal=False,
        k=5,
    )
    aggregate = aggregate_metrics([scorable_hit, scorable_miss, refusal, not_indexed], k=5)
    assert aggregate.total_cases == 4
    assert aggregate.scorable_case_count == 2
    assert aggregate.hit_count == 1
    assert aggregate.recall_at_k == 0.5
    assert aggregate.mrr == round((1.0 / 1.0) / 2, 6)
    assert aggregate.refusal_case_count == 1
    assert aggregate.quote_not_indexed_case_count == 1
    assert aggregate.retrieved_beyond_k_count == 0
    assert aggregate.tier_distribution["high"] == 2
    assert aggregate.tier_distribution["low"] == 2
    assert aggregate.low_tier_rate == round(2 / 4, 6)


def test_aggregate_counts_retrieved_beyond_k() -> None:
    beyond = compute_case_metrics(
        quotes=["price of related goods"],
        retrieved_texts=RETRIEVED,
        indexed_texts=RETRIEVED,
        scores=[0.9, 0.8],
        is_refusal=False,
        k=1,
    )
    aggregate = aggregate_metrics([beyond], k=1)
    assert aggregate.hit_count == 0
    assert aggregate.recall_at_k == 0.0
    assert aggregate.retrieved_beyond_k_count == 1


def test_aggregate_empty_is_zeroed() -> None:
    aggregate = aggregate_metrics([], k=5)
    assert aggregate.recall_at_k == 0.0
    assert aggregate.mrr == 0.0
    assert aggregate.low_tier_rate == 0.0


def test_quote_coverage_separates_retrieved_from_indexed() -> None:
    coverage = quote_coverage(
        ["complementary goods", "gross domestic product"],
        RETRIEVED[:1],
        RETRIEVED,
    )
    by_quote = {item.quote: item for item in coverage}
    assert by_quote["complementary goods"].found_in_indexed
    assert not by_quote["complementary goods"].found_in_retrieved
    assert not by_quote["gross domestic product"].found_in_indexed
