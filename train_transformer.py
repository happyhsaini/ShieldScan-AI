"""Train the three-class message detector and compare it with the TF-IDF baseline.

Workbook Word rows are treated as Suspicious signals; workbook sentence rows
retain their Fake label. Existing ham/spam data and curated examples supply all
three classes. Evaluation uses one stratified holdout shared by both models.
"""
import argparse
import json
import os
import random
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import BASE_DIR, RANDOM_STATE
from src.feature_engineering import build_vectorizer
from src.training_data import ID_TO_LABEL, LABEL_TO_ID, load_three_class_data
from src.utils import save_pickle

DEFAULT_DATA = os.path.join(BASE_DIR, "training_data", "fake_messages_dataset.xlsx")
DEFAULT_SUPPLEMENTAL = os.path.join(BASE_DIR, "training_data", "supplemental_messages.csv")
DEFAULT_MODEL = os.path.join(BASE_DIR, "transformer_model")
MAX_LENGTH = 128
BATCH_SIZE = 16


class MessageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.tokens = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.tokens["input_ids"][index],
            "attention_mask": self.tokens["attention_mask"][index],
            "labels": self.labels[index],
        }


def train_and_compare(data_path, baseline_path, supplemental_path, model_name, epochs, archive_path=None):
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    data = load_three_class_data(data_path, baseline_path, supplemental_path, archive_path)
    texts = data["message"].tolist()
    labels = data["label"].astype(int).tolist()
    class_ids = [LABEL_TO_ID[name] for name in LABEL_TO_ID]
    class_names = [ID_TO_LABEL[index] for index in class_ids]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels,
    )
    print(f"Stratified holdout: {len(train_texts)} train / {len(test_texts)} test")

    print("Training three-class TF-IDF + Logistic Regression baseline...")
    vectorizer = build_vectorizer()
    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)
    tfidf_model = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
        solver="lbfgs", C=1.0,
    )
    tfidf_model.fit(train_vectors, train_labels)
    tfidf_predictions = tfidf_model.predict(test_vectors).tolist()
    tfidf_accuracy = float(accuracy_score(test_labels, tfidf_predictions))

    print(f"Loading local pretrained encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    id2label = {index: ID_TO_LABEL[index] for index in class_ids}
    label2id = {name: index for index, name in id2label.items()}
    transformer = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    train_loader = DataLoader(
        MessageDataset(train_texts, train_labels, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        MessageDataset(test_texts, test_labels, tokenizer),
        batch_size=BATCH_SIZE,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer.to(device)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    counts = np.bincount(np.asarray(train_labels), minlength=3)
    # Square-root balancing helps the smaller review class without overwhelming
    # the much larger real-message class with a few repeated examples.
    weights = np.sqrt(len(train_labels) / (3 * np.maximum(counts, 1)))
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print("Training class counts:", {ID_TO_LABEL[i]: int(counts[i]) for i in class_ids})

    for epoch in range(epochs):
        transformer.train()
        losses = []
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = transformer(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, batch["labels"], weight=class_weights,
                label_smoothing=0.02,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            if batch_index % 25 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: batch {batch_index}/{len(train_loader)}", flush=True)
        print(f"Epoch {epoch + 1}/{epochs}: mean loss {np.mean(losses):.4f}", flush=True)

    transformer.eval()
    transformer_predictions = []
    with torch.inference_mode():
        for batch in test_loader:
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            predictions = transformer(**inputs).logits.argmax(dim=-1)
            transformer_predictions.extend(predictions.cpu().tolist())

    transformer_accuracy = float(accuracy_score(test_labels, transformer_predictions))
    difference_pp = (transformer_accuracy - tfidf_accuracy) * 100
    winner = "DistilBERT Transformer" if difference_pp > 0 else ("TF-IDF" if difference_pp < 0 else "Tie")
    tfidf_report = classification_report(
        test_labels, tfidf_predictions, labels=class_ids, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    transformer_report = classification_report(
        test_labels, transformer_predictions, labels=class_ids, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    print("\nTF-IDF held-out classification report:")
    print(classification_report(test_labels, tfidf_predictions, labels=class_ids, target_names=class_names, zero_division=0))
    print("Transformer held-out classification report:")
    print(classification_report(test_labels, transformer_predictions, labels=class_ids, target_names=class_names, zero_division=0))

    comparison = {
        "model": "DistilBERT (three-class encoder-only Transformer)",
        "model_name": model_name,
        "class_names": class_names,
        "class_distribution": {name: int((data["label_name"] == name).sum()) for name in class_names},
        "epochs": epochs,
        "split": "stratified 80/20 holdout",
        "random_state": RANDOM_STATE,
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "tfidf_accuracy_pct": round(tfidf_accuracy * 100, 2),
        "transformer_accuracy_pct": round(transformer_accuracy * 100, 2),
        "accuracy_difference_pp": round(difference_pp, 2),
        "winner": winner,
        "tfidf_macro_f1_pct": round(float(f1_score(test_labels, tfidf_predictions, labels=class_ids, average="macro", zero_division=0)) * 100, 2),
        "transformer_macro_f1_pct": round(float(f1_score(test_labels, transformer_predictions, labels=class_ids, average="macro", zero_division=0)) * 100, 2),
        "tfidf_f1_fake": round(float(tfidf_report["Fake"]["f1-score"]) * 100, 2),
        "transformer_f1_fake": round(float(transformer_report["Fake"]["f1-score"]) * 100, 2),
        "tfidf_f1_suspicious": round(float(tfidf_report["Suspicious"]["f1-score"]) * 100, 2),
        "transformer_f1_suspicious": round(float(transformer_report["Suspicious"]["f1-score"]) * 100, 2),
        "tfidf_precision_fake": round(float(precision_score(test_labels, tfidf_predictions, labels=[LABEL_TO_ID["Fake"]], average="macro", zero_division=0)) * 100, 2),
        "transformer_precision_fake": round(float(precision_score(test_labels, transformer_predictions, labels=[LABEL_TO_ID["Fake"]], average="macro", zero_division=0)) * 100, 2),
        "tfidf_recall_fake": round(float(recall_score(test_labels, tfidf_predictions, labels=[LABEL_TO_ID["Fake"]], average="macro", zero_division=0)) * 100, 2),
        "transformer_recall_fake": round(float(recall_score(test_labels, transformer_predictions, labels=[LABEL_TO_ID["Fake"]], average="macro", zero_division=0)) * 100, 2),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # Fit the deployable TF-IDF baseline on every labeled row after evaluation.
    final_vectorizer = build_vectorizer()
    full_vectors = final_vectorizer.fit_transform(texts)
    final_tfidf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
        solver="lbfgs", C=1.0,
    )
    final_tfidf.fit(full_vectors, labels)
    save_pickle(final_tfidf, os.path.join(BASE_DIR, "model.pkl"))
    save_pickle(final_vectorizer, os.path.join(BASE_DIR, "vectorizer.pkl"))
    save_pickle({
        "accuracy": transformer_accuracy,
        "tfidf_accuracy": tfidf_accuracy,
        "macro_f1": float(f1_score(test_labels, transformer_predictions, labels=class_ids, average="macro", zero_division=0)),
        "features": full_vectors.shape[1],
        "training_samples": len(data),
        "test_samples": len(test_texts),
        "class_distribution": comparison["class_distribution"],
    }, os.path.join(BASE_DIR, "model_info.pkl"))

    model_dir = os.path.join(BASE_DIR, "transformer_model")
    os.makedirs(model_dir, exist_ok=True)
    transformer.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    with open(os.path.join(BASE_DIR, "model_comparison.json"), "w", encoding="utf-8") as output:
        json.dump(comparison, output, indent=2, ensure_ascii=False)

    print(f"Saved all-row TF-IDF model: {os.path.join(BASE_DIR, 'model.pkl')}")
    print(f"Saved fine-tuned three-class encoder: {model_dir}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA, help="Labeled Excel source workbook")
    parser.add_argument("--baseline", default=os.path.join(BASE_DIR, "combined_data.csv"))
    parser.add_argument("--supplemental", default=DEFAULT_SUPPLEMENTAL)
    parser.add_argument("--archive", default=os.path.join(BASE_DIR, "training_data", "archive_spam.zip"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    train_and_compare(args.data, args.baseline, args.supplemental, args.model, args.epochs, args.archive)
