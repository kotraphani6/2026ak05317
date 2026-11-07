
# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and compare
five supervised machine learning classification models
for predicting whether a breast tumour is Benign or
Malignant using the Breast Cancer Wisconsin (Diagnostic)
dataset.

The project implements an interactive Streamlit web
application that allows users to upload a testing dataset,
select a classification algorithm and evaluate its
performance using standard classification metrics.

## 2. Dataset Description

Dataset: Breast Cancer Wisconsin (Diagnostic)

Source: UCI Machine Learning Repository

Dataset DOI: https://doi.org/10.24432/C5DW2B

The dataset contains:
- 569 observations
- 30 numerical input features
- 1 target variable (Diagnosis)
- 357 Benign observations
- 212 Malignant observations
- No missing values in the original dataset

The features describe characteristics of cell nuclei,
including radius, texture, perimeter, area, smoothness,
compactness, concavity and symmetry.

The target classes are:
- Benign
- Malignant

## 3. Train-Test Splitting

The dataset was divided using stratified sampling:

- Training data: 80% (455 observations)
- Testing data: 20% (114 observations)
- Random state: 42

Feature scaling was performed using StandardScaler
within machine learning pipelines wherever required.

## 4. Models Used

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 5. Evaluation Metrics

The models were evaluated using:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

Malignant was treated as the positive class for
Precision, Recall and F1-score.

## 6. Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| SVM | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## 7. Model Performance Observations

### Logistic Regression
Achieved 96.49% accuracy and the highest AUC of
0.9960, demonstrating excellent classification ability.

### Decision Tree
Achieved 92.98% accuracy. It provides interpretable
classification rules but showed comparatively lower
performance.

### KNN
Achieved 95.61% accuracy and 93.83% F1-score.
It demonstrated strong classification performance.

### Naive Bayes
Achieved 92.11% accuracy and 0.9891 AUC.
Its recall was comparatively lower.

### Support Vector Machine
Achieved the highest accuracy of 97.37%, precision of
100%, F1-score of 96.30% and MCC of 0.9442.

### Overall Winner
Support Vector Machine was the best overall performer
on the independent test set based on accuracy,
precision, F1-score and MCC. Logistic Regression
achieved the highest AUC.

Further repeated stratified cross-validation is
recommended to assess model stability.

## 8. Streamlit Application Features

The application provides:

- CSV dataset upload.
- Classification model selection.
- Evaluation metrics display.
- Confusion matrix visualization.
- Classification report.
- Downloadable prediction results.

## 9. Repository Structure

```text
Breast_Cancer_Classification_Project/
|
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|
|-- model/
|   |-- logistic_regression.pkl
|   |-- decision_tree.pkl
|   |-- knn.pkl
|   |-- naive_bayes.pkl
|   |-- support_vector_machine.pkl
|
|-- results/
|   |-- final_model_evaluation.csv
|   |-- model_observations.csv
|   |-- confusion_matrices.png
```

## 10. Installation and Execution

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in a browser.

## 11. Deployment

The application is intended to be deployed using
Streamlit Community Cloud.

Deployment URL: To be added after deployment.

GitHub Repository: To be added after repository creation.

## 12. Conclusion

The project demonstrates the implementation and
comparison of five supervised machine learning
classification algorithms.

Among the evaluated models, Support Vector Machine
achieved the strongest overall test-set performance.
The interactive Streamlit application provides an
accessible interface for evaluating the saved models.

## 13. Disclaimer

This project is developed for academic and educational
purposes only. It is not intended for medical diagnosis
or clinical decision-making.
