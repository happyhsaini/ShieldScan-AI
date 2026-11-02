"""Train and evaluate the three-class TF-IDF detector."""
import json
import os
from datetime import datetime, timezone
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.config import BASE_DIR, RANDOM_STATE
from src.feature_engineering import build_vectorizer
from src.training_data import ID_TO_LABEL, LABEL_TO_ID, load_three_class_data
from src.utils import save_pickle

DATA = os.path.join(BASE_DIR, "training_data", "fake_messages_dataset.xlsx")
BASELINE = os.path.join(BASE_DIR, "combined_data.csv")
SUPPLEMENTAL = os.path.join(BASE_DIR, "training_data", "supplemental_messages.csv")
ARCHIVE_ZIP = os.path.join(BASE_DIR, "training_data", "archive_spam.zip")

def main():
    df = load_three_class_data(DATA, BASELINE, SUPPLEMENTAL, ARCHIVE_ZIP)
    texts, labels = df.message.to_numpy(), df.label.to_numpy()
    ids = list(LABEL_TO_ID.values())
    names = [ID_TO_LABEL[i] for i in ids]
    xtr, xte, ytr, yte = train_test_split(texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels)
    vec = build_vectorizer()
    xtrv, xtev = vec.fit_transform(xtr), vec.transform(xte)
    mdl = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs", C=1.0)
    mdl.fit(xtrv, ytr)
    raw_probabilities = mdl.predict_proba(xtev)
    fake_probability = raw_probabilities[:, list(mdl.classes_).index(LABEL_TO_ID["Fake"])]
    suspicious_probability = raw_probabilities[:, list(mdl.classes_).index(LABEL_TO_ID["Suspicious"])]
    threat_score = fake_probability + 0.89 * suspicious_probability
    threat_score = np.where(suspicious_probability >= fake_probability, np.minimum(threat_score, 0.949999), threat_score)
    pred = np.where(threat_score >= 0.95, LABEL_TO_ID["Fake"], np.where(threat_score >= 0.85, LABEL_TO_ID["Suspicious"], LABEL_TO_ID["Real"]))
    report = classification_report(yte, pred, labels=ids, target_names=names, output_dict=True, zero_division=0)
    print("Three-class TF-IDF held-out report using live threat-score bands (score=Fake+0.89*Suspicious; Fake >=95%, Suspicious 85-<95%, Real <85%):")
    print(classification_report(yte, pred, labels=ids, target_names=names, digits=4, zero_division=0))
    print("Confusion matrix (rows actual, columns predicted; Real, Suspicious, Fake):")
    print(confusion_matrix(yte, pred, labels=ids))
    counts = {ID_TO_LABEL[int(k)]: int(v) for k, v in df.label.value_counts().items()}
    info = {"accuracy": float(accuracy_score(yte, pred)), "features": int(xtrv.shape[1]), "training_samples": len(xtr), "test_samples": len(xte), "class_distribution": counts, "class_names": names, "training_mode": "three_class_tfidf", "threat_score_bands": {"real_below": 85, "suspicious_from": 85, "fake_from": 95}}
    save_pickle(mdl, os.path.join(BASE_DIR, "model.pkl"))
    save_pickle(vec, os.path.join(BASE_DIR, "vectorizer.pkl"))
    save_pickle(info, os.path.join(BASE_DIR, "model_info.pkl"))
    comparison = {"model": "TF-IDF + Logistic Regression (active three-class model)", "source_archive": "training_data/archive_spam.zip", "threat_score_bands": {"score_formula": "Fake + 0.89 * Suspicious", "Real": "0-<85%", "Suspicious": "85-<95%", "Fake": "95-100%"}, "epochs": 0, "split": "stratified 80/20 holdout", "random_state": RANDOM_STATE, "train_samples": len(xtr), "test_samples": len(xte), "accuracy_pct": round(info["accuracy"] * 100, 2), "evaluation_policy": "Threat score = Fake + 0.89 * Suspicious; Suspicious-led outputs are capped below 95%; score >=95% is Fake, 85-<95% Suspicious, below 85% Real.", "classes": names, "class_distribution": counts, "class_metrics": {name: {key: round(float(report[name][key]) * 100, 2) for key in ("precision", "recall", "f1-score")} for name in names}, "transformer_status": "Previous checkpoint remains binary and is not used for live predictions until three-class fine-tuning completes.", "evaluated_at": datetime.now(timezone.utc).isoformat()}
    with open(os.path.join(BASE_DIR, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"Saved three-class TF-IDF model: {len(xtr)} train / {len(xte)} test; rows={len(df)}; classes={counts}")

if __name__ == "__main__": main()
