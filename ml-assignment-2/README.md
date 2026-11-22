# ShopSense — Predicting Purchase Intent of Online Shoppers

**ML Assignment 2 — M.Tech (AIML), BITS Pilani WILP**

- **Live Streamlit app:** `<paste your Streamlit Community Cloud URL here>`
- **GitHub repository:** https://github.com/kotraphani6/2026ak05317/tree/BITSMLASSIGNMENT2/ml-assignment-2

---

## a. Problem statement

An e-commerce site sees thousands of browsing sessions a day, but only a small
fraction end in a purchase. The goal is to build a **binary classifier** that
uses session behaviour (pages visited, time spent, bounce/exit rates, page
value, month, visitor type, etc.) to predict whether a session will generate
revenue (`Revenue = TRUE`). Such a model can drive real-time interventions —
targeted offers, chat prompts, or retargeting — for visitors likely to buy.

Five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM)
are trained on the same data, compared on six metrics, and served through an
interactive Streamlit app.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository (Sakar et al., 2018) — https://archive.ics.uci.edu/dataset/468 |
| Instances | 12,330 sessions (each from a different user over one year) |
| Input features | 17 (10 numeric, 7 categorical) |
| Target | `Revenue` — TRUE (purchase) / FALSE |
| Class balance | 10,422 FALSE (84.5%) vs 1,908 TRUE (15.5%) — imbalanced |
| Missing values | None |

**Features**

| Feature | Type | Meaning |
|---|---|---|
| Administrative, Informational, ProductRelated | numeric | Number of pages of each type visited in the session |
| Administrative_Duration, Informational_Duration, ProductRelated_Duration | numeric | Seconds spent on each page type |
| BounceRates | numeric | Avg. % of visitors who enter on the page and leave without another request |
| ExitRates | numeric | Avg. % of page views that were the last in the session |
| PageValues | numeric | Avg. value of the pages visited before a transaction |
| SpecialDay | numeric | Closeness of the visit to a special day (e.g. Mother's Day), 0–1 |
| Month | categorical | Month of the visit |
| OperatingSystems, Browser, Region, TrafficType | categorical (integer codes) | Technical / acquisition attributes |
| VisitorType | categorical | Returning, New, or Other |
| Weekend | boolean | Whether the visit was on a weekend |

**Pre-processing**

- Stratified 80/20 split (random_state = 2026): 9,864 training rows, 2,466 test rows.
  The test split is saved unchanged as `test_data.csv`.
- Numeric features → `StandardScaler`; categorical features (including the
  integer-coded OS/Browser/Region/TrafficType, which have no numeric order) →
  `OneHotEncoder(handle_unknown="ignore")`.
- Naive Bayes uses only the 10 numeric features after a Yeo-Johnson
  `PowerTransformer`. Gaussian NB assumes normally distributed features, and
  one-hot 0/1 columns plus heavily skewed durations break that assumption
  (accuracy fell to 0.23 with them included).
- `class_weight="balanced"` is used for Logistic Regression, Decision Tree and
  SVM to counter the 85/15 imbalance.
- Every model is saved as a full scikit-learn `Pipeline` (preprocessing +
  classifier), so the app feeds raw CSV columns straight in.

## c. GitHub repository link

https://github.com/kotraphani6/2026ak05317/tree/BITSMLASSIGNMENT2/ml-assignment-2

```
ml-assignment-2/
├── app.py                      # Streamlit app (inference only)
├── train_models.py             # trains, evaluates and saves all 5 models
├── shopper_prep.py             # shared column lists + cleaning used by both
├── requirements.txt
├── README.md
├── test_data.csv               # 20% hold-out split with target column
├── data/
│   └── online_shoppers_intention.csv   # original UCI file
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metrics.csv             # hold-out metrics table below
```

## d. Models used

| Model | Configuration |
|---|---|
| Logistic Regression | L2, C = 0.5, balanced class weights |
| Decision Tree | max_depth = 6, min_samples_leaf = 20, balanced class weights |
| K-Nearest Neighbours | k = 15, distance-weighted |
| Naive Bayes | Gaussian NB on power-transformed numeric features |
| Support Vector Machine | RBF kernel, C = 1, balanced class weights, probabilities via `CalibratedClassifierCV` |

### Comparison table (hold-out test set, 2,466 sessions)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8350 | 0.8932 | 0.4789 | 0.7435 | 0.5826 | 0.5039 |
| Decision Tree | 0.8560 | **0.9162** | 0.5225 | **0.8194** | **0.6381** | **0.5755** |
| KNN | 0.8719 | 0.8238 | **0.6854** | 0.3194 | 0.4357 | 0.4089 |
| Naive Bayes | 0.8114 | 0.8437 | 0.4376 | 0.7618 | 0.5559 | 0.4748 |
| Support Vector Machine | **0.8788** | 0.8972 | 0.6446 | 0.4843 | 0.5531 | 0.4912 |

Precision, Recall and F1 are for the positive class (`Revenue = TRUE`).
Because only 15.5% of sessions convert, a model that always predicts "no
purchase" would already score 0.845 accuracy. **AUC, F1 and MCC are therefore
the more meaningful metrics here.**

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, stable baseline (AUC 0.89). Balanced class weights push recall to 0.74, but precision is below 0.5, so about half of predicted buyers do not buy. A linear boundary cannot fully capture threshold effects such as "PageValues > 0", which limits its MCC to 0.50. |
| Decision Tree | Best overall (AUC 0.916, F1 0.638, MCC 0.576) and catches 82% of buyers. The data contains sharp, rule-like signals that axis-aligned splits model naturally: PageValues alone carries 79% of the tree's feature importance, followed by Month = Nov (8%) and ProductRelated_Duration. Depth 6 and a minimum of 20 samples per leaf keep it from overfitting, and the model stays interpretable. |
| KNN | Highest precision (0.69) but lowest recall (0.32) and the weakest AUC (0.82). Without class weighting, the majority class dominates most neighbourhoods. Distances across 70+ one-hot and scaled dimensions are also less informative, so KNN is conservative and misses most buyers. |
| Naive Bayes | Usable only after removing one-hot features and power-transforming the skewed numerics: the feature-independence and Gaussian assumptions are clearly violated (e.g. page counts and durations are strongly correlated). After that it reaches recall 0.76 and MCC 0.47, but it has the lowest accuracy and precision. It is fast and needs no tuning, so it suits a quick baseline. |
| Support Vector Machine | Highest accuracy (0.879) and a high AUC (0.897). After probability calibration its default 0.5 threshold is conservative: precision 0.64, recall 0.48. Its ranking quality is close to the Decision Tree's, so a lower decision threshold would trade precision for recall. It is the slowest model to train. |
| **Overall, winner for this dataset** | **Decision Tree.** It leads on AUC, Recall, F1 and MCC, the metrics that matter for an imbalanced target. It is also explainable to business users. SVM is the runner-up on ranking quality, and KNN suits cases where false alarms are costly. |

## Streamlit app features

- **CSV upload** in the sidebar (the bundled `test_data.csv` is used until a
  file is uploaded; a download button provides it as a template).
- **Model selection dropdown** for the five classifiers.
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC.
- **Confusion matrix**, **ROC curve** and **classification report**.
- **Compare tab** with all five models scored on the uploaded data (best value per metric highlighted) and a bar chart for any chosen metric.
- Downloadable predictions with purchase probabilities.
- Models are loaded from `model/*.joblib`; **nothing is retrained in the app**.
  The uploaded CSV must contain the 17 feature columns plus `Revenue`.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py        # optional: regenerates model/ and test_data.csv
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

1. Code lives in `ml-assignment-2/` on branch `BITSMLASSIGNMENT2` of `kotraphani6/2026ak05317`.
2. Go to https://streamlit.io/cloud, sign in with GitHub, and click **New app**.
3. Select the repository, branch `BITSMLASSIGNMENT2`, and main file path `ml-assignment-2/app.py`, then click **Deploy**.

The package versions in `requirements.txt` are pinned so that the pickled
pipelines load with the same scikit-learn version they were trained with.
