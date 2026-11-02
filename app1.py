"""
ShieldScan AI — Entry Point
Run from the project folder:
    python app1.py
"""
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Model file paths
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

if not os.path.isfile(MODEL_PATH) or not os.path.isfile(VECTORIZER_PATH):
    print("\n" + "=" * 60)
    print("  ❌ MODEL FILES NOT FOUND")
    print("=" * 60)
    print(f"Expected model at: {MODEL_PATH}")
    print(f"Expected vectorizer at: {VECTORIZER_PATH}")
    print("\nMake sure these files exist in the project root:")
    print("  - model.pkl")
    print("  - vectorizer.pkl")
    print("=" * 60 + "\n")
    sys.exit(1)

from src.app import app

if __name__ == "__main__":
    print("\n✅ Model files found.")
    port = int(os.environ.get("PORT", 5500))
    app.run(host="0.0.0.0", port=port, debug=False)