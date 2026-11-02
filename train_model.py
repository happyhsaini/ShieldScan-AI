"""
Train the Fake Message Detection model.

Run from project root folder (where combined_data.csv is):
    python train_model.py

If your CSV has a different name:
    python train_model.py --data mydata.csv
"""
import sys
import os
import argparse

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocess          import load_and_prepare_data
from src.feature_engineering import build_vectorizer
from src.utils               import save_pickle

# Paths — always relative to this file
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV     = os.path.join(BASE_DIR, "combined_data.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
MODEL_INFO_PATH = os.path.join(BASE_DIR, "model_info.pkl")


def train(data_path: str):
    print("=" * 62)
    print("  FAKE MESSAGE DETECTION — MODEL TRAINING")
    print("=" * 62)

    # Validate CSV
    if not os.path.isfile(data_path):
        print(f"\n❌ ERROR: File not found: {data_path}")
        print("   Make sure combined_data.csv is in the SAME FOLDER as train_model.py")
        print("   Or use:  python train_model.py --data yourfile.csv")
        sys.exit(1)

    # Load
    print(f"\n📂 Loading: {data_path}")
    df = load_and_prepare_data(data_path)
    print(f"\n✓ Rows loaded: {len(df)}")
    vc = df["label"].value_counts()
    print(f"  Real (0): {vc.get(0,0)}  |  Fake (1): {vc.get(1,0)}")

    if len(df) < 20:
        print("\n❌ Need at least 20 rows to train."); sys.exit(1)

    # Split
    X, y = df["message"].values, df["label"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n✓ Train: {len(Xtr)}  |  Test: {len(Xte)}")

    # Vectorise
    print("\n📝 TF-IDF vectorising…")
    vec = build_vectorizer()
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)
    print(f"  Features: {Xtr_v.shape[1]}")

    # Train
    print("\n🚀 Training Logistic Regression…")
    mdl = LogisticRegression(max_iter=2000, class_weight="balanced",
                              random_state=42, solver="lbfgs", C=1.0)
    mdl.fit(Xtr_v, ytr)

    # Evaluate
    ypred = mdl.predict(Xte_v)
    acc   = accuracy_score(yte, ypred)
    print(f"\n📊 Test Accuracy: {acc*100:.2f}%")
    print(classification_report(yte, ypred, target_names=["Real","Fake"], digits=4))
    cm = confusion_matrix(yte, ypred)
    print(f"Confusion Matrix:\n  Real predicted as Real: {cm[0][0]}  as Fake: {cm[0][1]}")
    print(f"  Fake predicted as Real: {cm[1][0]}  as Fake: {cm[1][1]}")

    # Save
    save_pickle(mdl, MODEL_PATH)
    save_pickle(vec, VECTORIZER_PATH)
    save_pickle({"accuracy":acc,"features":Xtr_v.shape[1],
                 "training_samples":len(Xtr),"test_samples":len(Xte)}, MODEL_INFO_PATH)

    print(f"\n✅ Files saved in: {BASE_DIR}")
    print("   model.pkl  |  vectorizer.pkl  |  model_info.pkl")
    print("\n🎉 Done! Now run:  python app1.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_CSV,
                   help="Path to training CSV (default: combined_data.csv)")
    args = p.parse_args()
    train(args.data)
