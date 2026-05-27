from __future__ import annotations

from scripts.utils.topics import HINDI_STOPWORDS, extract_topics


def test_extract_topics_empty() -> None:
    assert extract_topics("") == []


def test_extract_topics_whitespace_only() -> None:
    assert extract_topics("   ") == []


def test_extract_topics_english() -> None:
    text = "machine learning neural network deep learning gradient descent backpropagation"
    topics = extract_topics(text, top_n=5)
    assert len(topics) > 0
    combined = " ".join(topics)
    assert any(kw in combined for kw in ["machine", "neural", "learning", "gradient"])


def test_extract_topics_hindi_stopwords_filtered() -> None:
    text = "yeh function ka return type string hai aur variable ko assign karo"
    topics = extract_topics(text, top_n=10)
    for t in topics:
        for word in t.split():
            assert word not in HINDI_STOPWORDS, f"Stopword '{word}' found in topics"


def test_extract_topics_top_n_respected() -> None:
    text = "apple banana cherry date elderberry fig grape honeydew kiwi lemon mango"
    topics = extract_topics(text, top_n=3)
    assert len(topics) <= 3


def test_extract_topics_returns_strings() -> None:
    text = "function return variable type assignment loop condition"
    topics = extract_topics(text)
    assert all(isinstance(t, str) for t in topics)


def test_hindi_stopwords_nonempty() -> None:
    assert len(HINDI_STOPWORDS) > 10


def test_extract_topics_single_word() -> None:
    topics = extract_topics("recursion", top_n=5)
    assert "recursion" in topics
