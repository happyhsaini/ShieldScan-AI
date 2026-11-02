# ShieldScan AI — Fake Message Detection Platform

## Quick Start (VS Code)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (for image analysis)
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
- **Ubuntu/WSL**: `sudo apt install tesseract-ocr`
- **Mac**: `brew install tesseract`

### 3. Train the model (first time only)
```bash
python train_model.py
```
This reads a local `combined_data.csv` and creates `model.pkl`, `vectorizer.pkl`, `model_info.pkl`. Training datasets are kept out of this public repository; provide your own CSV using the format below.

### 4. Run the website
```bash
python app1.py
```
Open: http://127.0.0.1:5500

---

## File Structure
```
project/
├── app1.py                  ← Entry point (run this)
├── train_model.py           ← Standalone training script
├── combined_data.csv        ← Local training dataset (not included)
├── model.pkl                ← Trained model (auto-generated)
├── vectorizer.pkl           ← TF-IDF vectorizer (auto-generated)
├── model_info.pkl           ← Model metadata (auto-generated)
├── requirements.txt
├── src/
│   ├── app.py               ← All Flask routes
│   ├── config.py            ← Paths and thresholds
│   ├── preprocess.py        ← Text cleaning + CSV loading
│   ← feature_engineering.py ← TF-IDF builder
│   ├── explain.py           ← Human-readable explanations
│   ├── ocr.py               ← Image → text (Tesseract)
│   └── utils.py             ← pickle helpers
├── templates/
│   └── index.html           ← Full UI
└── static/
    ├── css/style.css
    └── js/script.js
```

## CSV Format for Training
| Column | Values |
|--------|--------|
| label / v1 | `ham` or `spam` (or `real`/`fake`) |
| text / v2 / message | message text |

## Prediction Logic
| Threat score | Label |
|-----------------|-------|
| < 85% | ✅ Real |
| 85–<95% | ⚠ Suspicious |
| ≥95% | 🚫 Fake |

Threat score = fake probability + 0.89 × suspicious probability. Scores below 85% are Real, 85–<95% are Suspicious, and 95% or higher are Fake. Suspicious-led predictions are capped below 95%.

## TF-IDF vs Transformer accuracy

The dashboard compares both classifiers on the same stratified 80/20 holdout so
accuracy changes are measured on identical test messages. Install the optional
training libraries and run the comparison:

```bash
pip install -r requirements-transformer.txt
python train_transformer.py
```

The script fine-tunes the pretrained DistilBERT encoder for fake-message
classification, using tokenization, learned token/position embeddings from the
pretrained model, a warmup schedule, class-balanced loss, and label smoothing. It
writes `model_comparison.json` and the fine-tuned weights to `transformer_model/`. Transformer weights are git-ignored because the model exceeds GitHub's standard file-size limit; the committed app uses the included TF-IDF model by default.
After training, restart the Flask app and open the Dashboard to see TF-IDF accuracy,
Transformer accuracy, and their percentage-point difference. When `transformer_model/`
exists and optional dependencies are installed, DistilBERT handles live predictions;
otherwise the app falls back to TF-IDF. A positive value means
DistilBERT scored higher; a negative value means TF-IDF scored higher. Accuracy is
measured on this dataset's held-out split and is not a guarantee of real-world results.
