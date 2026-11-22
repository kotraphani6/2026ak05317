"""Train five classifiers on the UCI Online Shoppers Purchasing Intention data.

Outputs:
  model/<name>.joblib   - full pipeline (preprocessing + classifier)
  model/metrics.csv     - hold-out metrics for all five models
  test_data.csv         - the untouched 20% hold-out split (raw columns + Revenue)

Run:  python train_models.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import (OneHotEncoder, PowerTransformer,
                                   StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from shopper_prep import (CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS,
                          TARGET, clean_frame)

ROOT = Path(__file__).parent
RAW_CSV = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
SEED = 2026


def make_preprocessor(model_name: str) -> ColumnTransformer:
    if model_name == "Naive Bayes":
        # Gaussian NB assumes bell-shaped features: one-hot 0/1 columns and the
        # heavily right-skewed durations break that assumption badly, so NB
        # uses only the numeric columns after a Yeo-Johnson power transform.
        return ColumnTransformer([("num", PowerTransformer(), NUMERIC_COLS)])
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])


CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", C=0.5),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=20, class_weight="balanced",
        random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    "Naive Bayes": GaussianNB(),
    # SVC has no native probabilities; calibrate it so AUC can be computed
    "Support Vector Machine": CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            random_state=SEED),
        ensemble=False),
}


def file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def score(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    raw = pd.read_csv(RAW_CSV)
    print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns")

    train_raw, test_raw = train_test_split(
        raw, test_size=0.20, stratify=raw[TARGET], random_state=SEED)
    test_raw.to_csv(ROOT / "test_data.csv", index=False)

    train_df, test_df = clean_frame(train_raw), clean_frame(test_raw)
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET]
    print(f"Train {len(X_train)} | Test {len(X_test)} | "
          f"positive rate {y_train.mean():.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    rows = []
    for name, clf in CLASSIFIERS.items():
        pipe = Pipeline([
            ("prep", make_preprocessor(name)),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        rows.append({"Model": name, **score(y_test, y_pred, y_prob)})
        joblib.dump(pipe, MODEL_DIR / file_name(name), compress=3)
        print(f"  trained + saved {name}")

    metrics = pd.DataFrame(rows).set_index("Model").round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv")
    print("\n" + metrics.to_string())


if __name__ == "__main__":
    main()
