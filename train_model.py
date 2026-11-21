from pathlib import Path
import io
import zipfile
import urllib.request
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "SMSSpamCollection"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_dataset():
    """Download and extract the official UCI SMS Spam Collection if not present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        return

    print("Dataset not found. Downloading the UCI SMS Spam Collection...")
    try:
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = next(name for name in archive.namelist() if name.endswith("SMSSpamCollection"))
            DATA_FILE.write_bytes(archive.read(member))
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset. Check your internet connection. "
            f"Dataset source: {DATA_URL}"
        ) from exc
    print(f"Dataset saved to: {DATA_FILE}")


def load_dataset():
    download_dataset()
    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1"
    )
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    return df


def train_and_save_model():
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    # TF-IDF is the required text feature extraction algorithm.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=["ham", "spam"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metrics = {
        "dataset_rows_after_duplicate_removal": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix_labels": ["ham", "spam"],
        "confusion_matrix": matrix.tolist()
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\\nTraining completed.")
    print(f"Messages used: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted; order: ham, spam):")
    print(matrix)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    return pipeline, metrics


if __name__ == "__main__":
    train_and_save_model()
