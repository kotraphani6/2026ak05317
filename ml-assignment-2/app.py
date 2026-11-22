"""ShopSense - purchase-intent classifier dashboard (Streamlit).

Loads the five pre-trained pipelines from model/ and evaluates them on an
uploaded test CSV. No model is retrained inside the app.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from shopper_prep import FEATURE_COLS, TARGET, clean_frame, missing_columns

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="ShopSense | Purchase Intent", page_icon="🛒",
                   layout="wide")


@st.cache_resource
def load_models() -> dict:
    return {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}


def evaluate(model, X, y) -> tuple[dict, pd.Series, pd.Series]:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    both_classes = y.nunique() == 2
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob) if both_classes else float("nan"),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pd.Series(pred, index=y.index), pd.Series(prob, index=y.index)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🛒 ShopSense")
st.sidebar.caption("Will this browsing session end in a purchase?")

uploaded = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
st.sidebar.download_button(
    "⬇️ Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
    file_name="test_data.csv", mime="text/csv")

model_name = st.sidebar.selectbox("Choose a classifier", list(MODEL_FILES))
threshold_note = st.sidebar.empty()

# ---------------------------------------------------------------- data
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    source = f"uploaded file **{uploaded.name}**"
else:
    raw = pd.read_csv(SAMPLE_CSV)
    source = "bundled **test_data.csv** (upload your own in the sidebar)"

st.title("Online Shopper Purchase-Intent Classifier")
st.markdown(f"Evaluating on {source} — {len(raw):,} sessions.")

missing = missing_columns(raw)
if missing:
    st.error("The CSV is missing required columns: " + ", ".join(missing))
    st.stop()

try:
    data = clean_frame(raw)
except Exception as exc:  # malformed values
    st.error(f"Could not parse the CSV: {exc}")
    st.stop()

if data[TARGET].isna().any():
    st.error(f"Column '{TARGET}' must contain only TRUE/FALSE or 1/0 values.")
    st.stop()

X, y = data[FEATURE_COLS], data[TARGET].astype(int)
models = load_models()

tab_single, tab_compare, tab_data = st.tabs(
    ["🔍 Selected model", "📊 Compare all models", "🗂️ Data preview"])

# ---------------------------------------------------------------- tab 1
with tab_single:
    metrics, pred, prob = evaluate(models[model_name], X, y)
    st.subheader(f"{model_name} — evaluation metrics")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No purchase", "Purchase"],
                    yticklabels=["No purchase", "Purchase"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**ROC curve**")
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, prob)
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            ax.plot(fpr, tpr, color="#1f77b4", lw=2,
                    label=f"AUC = {metrics['AUC']:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)
        else:
            st.info("ROC needs both classes present in the uploaded data.")

    st.markdown("**Classification report**")
    report = classification_report(
        y, pred, labels=[0, 1], target_names=["No purchase", "Purchase"],
        output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

    out = raw.copy()
    out["Predicted_Revenue"] = pred.astype(bool).values
    out["Purchase_Probability"] = prob.round(4).values
    st.download_button("⬇️ Download predictions", out.to_csv(index=False),
                       file_name=f"predictions_{MODEL_FILES[model_name][:-7]}.csv",
                       mime="text/csv")

# ---------------------------------------------------------------- tab 2
with tab_compare:
    st.subheader("All five models on this data")
    table = pd.DataFrame(
        {name: evaluate(m, X, y)[0] for name, m in models.items()}).T
    st.dataframe(table.style.format("{:.4f}").highlight_max(
        axis=0, color="#c7e9c0"), width="stretch")

    metric_pick = st.radio("Metric to plot", list(table.columns),
                           index=5, horizontal=True)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ordered = table[metric_pick].sort_values()
    ax.barh(ordered.index, ordered.values, color="#4c72b0")
    for i, v in enumerate(ordered.values):
        ax.text(v, i, f" {v:.3f}", va="center")
    ax.set_xlabel(metric_pick)
    ax.set_xlim(0, max(1.0, ordered.max() * 1.1))
    st.pyplot(fig)

# ---------------------------------------------------------------- tab 3
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(raw):,}")
    c2.metric("Input features", len(FEATURE_COLS))
    c3.metric("Purchase rate", f"{y.mean():.1%}")
    st.dataframe(raw.head(50), width="stretch")

st.caption("Models were trained offline (train_models.py) on 80% of the UCI "
           "Online Shoppers Purchasing Intention dataset; the app only runs "
           "inference.")
