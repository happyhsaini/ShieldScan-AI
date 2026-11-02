import re
import pandas as pd


# ── Text cleaner ───────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Normalise text for TF-IDF.
    Replaces URLs / emails / numbers with placeholder tokens so the model
    can still learn from their *presence* without memorising exact values.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URLTOKEN ", text)
    text = re.sub(r"\S+@\S+\.\S+", " EMAILTOKEN ", text)
    text = re.sub(r"\b\d{10,}\b", " PHONETOKEN ", text)      # phone numbers
    text = re.sub(r"\b\d+\b", " NUMTOKEN ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── CSV loader ─────────────────────────────────────────────────────────────────
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Supports column layouts:
      • v1, v2          (classic SMS spam dataset)
      • label, text
      • label, message
      • Any column whose name contains 'label' / 'text' / 'message'
    Labels accepted: ham / real / safe / 0 / legitimate → 0 (Real)
                     spam / fake / 1 / fraud / phishing  → 1 (Fake)
    """
    # 1. Read file ────────────────────────────────────────────────────────────
    try:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin-1")
    except Exception as e:
        raise Exception(f"Could not read CSV: {e}")

    original_cols = list(df.columns)
    cols_lower    = [str(c).strip().lower() for c in original_cols]

    # 2. Detect label column ──────────────────────────────────────────────────
    label_col = None
    for priority in [["v1"], ["label", "labels", "target", "class"]]:
        for i, col in enumerate(cols_lower):
            if col in priority:
                label_col = original_cols[i]
                break
        if label_col:
            break
    if label_col is None:
        for i, col in enumerate(cols_lower):
            if "label" in col:
                label_col = original_cols[i]
                break

    # 3. Detect text column ───────────────────────────────────────────────────
    text_col = None
    for priority in [["v2"], ["text", "message", "content", "body", "email"]]:
        for i, col in enumerate(cols_lower):
            if col in priority:
                text_col = original_cols[i]
                break
        if text_col:
            break
    if text_col is None:
        for i, col in enumerate(cols_lower):
            if any(x in col for x in ["text", "message", "content", "body"]):
                text_col = original_cols[i]
                break

    if label_col is None or text_col is None:
        raise Exception(
            f"Required columns not found. Found: {original_cols}. "
            f"Need a label column (v1/label) and a text column (v2/text/message)."
        )

    # 4. Clean & normalise ────────────────────────────────────────────────────
    df = df[[label_col, text_col]].copy()
    df.columns = ["label", "message"]
    df = df.dropna()
    df["label"]   = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str)
    df = df[df["message"].str.len() > 2]
    df = df.drop_duplicates()

    # 5. Map labels ───────────────────────────────────────────────────────────
    def map_label(x):
        x = str(x).strip().lower()
        if x in ("ham", "real", "safe", "0", "legitimate", "genuine"):
            return 0
        if x in ("spam", "fake", "1", "fraud", "phishing", "scam"):
            return 1
        return None

    df["label"] = df["label"].apply(map_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # 6. Clean message text ───────────────────────────────────────────────────
    df["message"] = df["message"].apply(clean_text)

    print(f"[preprocess] Loaded {len(df)} rows | "
          f"Real: {(df['label']==0).sum()} | Fake: {(df['label']==1).sum()}")
    return df
