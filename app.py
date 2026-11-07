
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Classification System")

st.markdown("""
### Machine Learning Model Evaluation

This application compares five supervised machine learning
classifiers using the Breast Cancer Wisconsin dataset.

Upload a testing CSV file, select a model and evaluate its
classification performance.
""")

st.divider()

# ----------------------------------------------------------
# MODEL CONFIGURATION
# ----------------------------------------------------------

MODEL_DIR = "model"

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

# ----------------------------------------------------------
# DATASET UPLOAD
# ----------------------------------------------------------

st.subheader("1. Upload Testing Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")

    except Exception as e:
        st.error(f"Unable to read CSV file: {e}")
        st.stop()

else:

    default_file = "test_data.csv"

    if os.path.exists(default_file):

        st.info(
            "No file uploaded. Using the default test_data.csv."
        )

        data = pd.read_csv(default_file)

    else:
        st.warning("Please upload a testing CSV file.")
        st.stop()

# ----------------------------------------------------------
# DATASET VALIDATION
# ----------------------------------------------------------

st.subheader("2. Dataset Preview")

st.dataframe(data.head(), use_container_width=True)

st.write("Dataset dimensions:", data.shape)

if "Diagnosis" not in data.columns:
    st.error(
        "The uploaded CSV must contain the Diagnosis target column."
    )
    st.stop()

# ----------------------------------------------------------
# MODEL SELECTION
# ----------------------------------------------------------

st.divider()

st.subheader("3. Select Classification Model")

selected_model = st.selectbox(
    "Choose a Machine Learning Algorithm",
    list(model_files.keys())
)

# ----------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------

if st.button("Evaluate Model", type="primary"):

    model_path = os.path.join(
        MODEL_DIR,
        model_files[selected_model]
    )

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    # Load saved model
    model = joblib.load(model_path)

    # Separate features and target
    X_test = data.drop(columns=["Diagnosis"])
    y_test = data["Diagnosis"]

    # Validate feature names
    expected_features = list(model.feature_names_in_)
    uploaded_features = list(X_test.columns)

    if set(expected_features) != set(uploaded_features):
        st.error(
            "The uploaded dataset does not contain the "
            "required feature columns."
        )
        st.write("Expected features:", expected_features)
        st.write("Uploaded features:", uploaded_features)
        st.stop()

    # Ensure correct feature order
    X_test = X_test[expected_features]

    # Validate values
    if X_test.isnull().any().any():
        st.error("The uploaded dataset contains missing feature values.")
        st.stop()

    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        st.error("The dataset contains invalid numerical values.")
        st.stop()

    # Predict
    y_pred = model.predict(X_test)

    # Malignant is the positive class
    positive_class = "Malignant"

    class_index = list(model.classes_).index(positive_class)

    y_probability = model.predict_proba(X_test)[:, class_index]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        (y_test == positive_class).astype(int),
        y_probability
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=positive_class,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=positive_class,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=positive_class,
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # ------------------------------------------------------
    # DISPLAY EVALUATION METRICS
    # ------------------------------------------------------

    st.divider()

    st.subheader("4. Model Evaluation Results")

    st.write("Selected Model:", selected_model)

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1-score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # ------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------

    st.divider()

    st.subheader("5. Confusion Matrix")

    labels = ["Benign", "Malignant"]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_xlabel("Predicted Diagnosis")
    ax.set_ylabel("Actual Diagnosis")
    ax.set_title(selected_model + " - Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    # ------------------------------------------------------
    # CLASSIFICATION REPORT
    # ------------------------------------------------------

    st.divider()

    st.subheader("6. Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

    # ------------------------------------------------------
    # DOWNLOAD PREDICTIONS
    # ------------------------------------------------------

    output = data.copy()

    output["Predicted_Diagnosis"] = y_pred
    output["Malignant_Probability"] = y_probability

    st.download_button(
        label="Download Prediction Results",
        data=output.to_csv(index=False),
        file_name="classification_predictions.csv",
        mime="text/csv"
    )

    st.success("Model evaluation completed successfully!")

st.divider()

st.caption(
    "Academic machine learning project | "
    "Breast Cancer Wisconsin (Diagnostic) Dataset"
)

st.caption(
    "This application is for educational purposes only "
    "and is not intended for clinical diagnosis."
)
