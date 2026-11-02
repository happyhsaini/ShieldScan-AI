from sklearn.feature_extraction.text import TfidfVectorizer
from .config import MAX_FEATURES


def build_vectorizer() -> TfidfVectorizer:
    """
    TF-IDF with unigrams + bigrams.
    min_df=2 removes extremely rare tokens (reduces noise).
    max_df=0.95 removes tokens present in almost every document.
    """
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
        token_pattern=r"\b\w{2,}\b",
        sublinear_tf=True,          # log-scale TF helps with long messages
    )
