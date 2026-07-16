from __future__ import annotations

HINDI_STOPWORDS: frozenset[str] = frozenset(
    {
        "ab", "agar", "aur", "aap", "bhi", "char", "do", "ek",
        "hain", "hai", "hoga", "hogi", "hoge", "ho", "hum",
        "iska", "iski", "iske", "jab", "ka", "kab", "kaise",
        "kar", "karna", "karte", "karti", "karta", "ke", "ki",
        "ko", "kya", "lekin", "main", "mat", "mein", "mera",
        "meri", "mere", "nahin", "nahi", "nhi", "ne", "paanch",
        "par", "phir", "se", "tab", "teen", "tera", "teri",
        "tere", "toh", "tum", "uska", "uski", "uske", "wala",
        "wali", "wale", "woh", "ya", "yeh",
        "tha", "thi", "the",
    }
)


def extract_topics(text: str, top_n: int = 10) -> list[str]:
    if not text.strip():
        return []
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    except ImportError:
        return []

    combined_stopwords = list(HINDI_STOPWORDS | set(ENGLISH_STOP_WORDS))
    vectorizer = TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),
        stop_words=combined_stopwords,
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Zऀ-ॿ]{3,}\b",
    )
    try:
        matrix = vectorizer.fit_transform([text])
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    scores = matrix.toarray()[0]
    top_indices = scores.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_indices if scores[i] > 0]
