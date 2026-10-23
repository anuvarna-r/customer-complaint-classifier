from pathlib import Path
import json

import joblib
import pandas as pd
import sklearn

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "complaints_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"


def main():
    if not DATA_PATH.exists():
        print("Dataset not found:", DATA_PATH)
        print("Run data_cleaning.py first.")
        return

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(
        DATA_PATH,
        usecols=["product", "narrative"],
    ).dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        df["narrative"],
        df["product"],
        test_size=0.20,
        random_state=42,
        stratify=df["product"],
    )

    print("Training complaints:", len(X_train), flush=True)
    print("Testing complaints:", len(X_test), flush=True)

    # The vocabulary is learned from training data only.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=40000,
                ngram_range=(1, 2),
                min_df=3,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining model. Please wait...", flush=True)
    model.fit(X_train, y_train)

    print("Evaluating model...", flush=True)
    predictions = model.predict(X_test)

    # Baseline: always predict the most common category.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train.to_frame(), y_train)
    baseline_predictions = baseline.predict(X_test.to_frame())

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro")
        ),
        "baseline_accuracy": float(
            accuracy_score(y_test, baseline_predictions)
        ),
        "baseline_macro_f1": float(
            f1_score(
                y_test,
                baseline_predictions,
                average="macro",
            )
        ),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "random_state": 42,
        "scikit_learn_version": sklearn.__version__,
    }

    print("\nMODEL RESULTS")
    print("-------------")
    print(f"Baseline accuracy: {metrics['baseline_accuracy']:.4f}")
    print(f"Model accuracy:    {metrics['accuracy']:.4f}")
    print(f"Model Macro F1:    {metrics['macro_f1']:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / "classification_report.csv"
    )

    labels = model.classes_

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="Actual"),
        columns=pd.Index(labels, name="Predicted"),
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")

    with open(
        REPORT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    joblib.dump(
        model,
        MODEL_DIR / "complaint_classifier.joblib",
    )

    print("\nModel and performance reports saved successfully.")


if __name__ == "__main__":
    main()