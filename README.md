# Multiclass Obesity Level Classification

A machine learning project that classifies individuals into seven obesity levels based on eating habits, physical condition, and demographic features, using data from Mexico, Peru, and Colombia. The project compares a One-vs-Rest Logistic Regression model against a Sequential Neural Network, with full hyperparameter tuning, model explainability via SHAP, and feature importance analysis via LASSO.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Feature Selection](#3-feature-selection)
  - [4. Train-Test Split](#4-train-test-split)
  - [5. Model Training](#5-model-training)
  - [6. Hyperparameter Tuning](#6-hyperparameter-tuning)
  - [7. Model Explainability](#7-model-explainability)
- [Results](#results)
- [Conclusion](#conclusion)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)

---

## Problem Statement

Estimate an individual's obesity level based on lifestyle and physical attributes. This is a **7-class classification** problem with the following target labels:

| Class | Label |
|---|---|
| 0 | Insufficient Weight |
| 1 | Normal Weight |
| 2 | Overweight Level I |
| 3 | Overweight Level II |
| 4 | Obesity Type I |
| 5 | Obesity Type II |
| 6 | Obesity Type III |

---

## Dataset

**Obesity Levels Dataset** — individuals from Mexico, Peru, and Colombia.

| Property | Details |
|---|---|
| Samples | 2,111 |
| Features | 17 (before preprocessing) |
| Target | `NObeyesdad` — 7-class obesity level |
| Class balance | Near-equal distribution; balanced using the SMOTE filter in Weka by original researchers |

**Categorical Features:** Gender, `family_history_with_overweight`, `FAVC` (frequent high-caloric food), `CAEC` (food between meals), `SMOKE`, `SCC` (calorie monitoring), `CALC` (alcohol consumption), `MTRANS` (transportation mode)

**Numerical Features:** Age, Height, Weight, `FCVC` (vegetable frequency), `NCP` (main meals per day), `CH2O` (daily water intake), `FAF` (physical activity frequency), `TUE` (screen/technology use time)

---

## Project Pipeline

### 1. Data Preprocessing

Categorical features were encoded using `LabelEncoder`, then merged with the numerical features into a single dataframe. Numerical features were scaled using `StandardScaler` via a `ColumnTransformer` to avoid data leakage — the scaler was applied only to the 8 continuous columns while categorical columns were passed through unchanged.

### 2. Exploratory Data Analysis

Several EDA steps were performed to understand the data prior to modelling:

**Class distribution:** A bar chart confirmed the near-equal class distribution across all seven obesity levels, validating the SMOTE balancing performed by the original researchers.

**Categorical distributions:** Count plots were generated for all categorical variables. Gender and the target class `NObeyesdad` showed equal distribution, while variables like `CAEC` and `MTRANS` showed meaningful variation across categories.

**Continuous distributions:** Histograms with KDE overlays revealed that Height and Weight are approximately normally distributed, while Age is positively skewed. Q-Q plots were used to assess normality more formally.

**Skewness & Kurtosis:** Numerical assessment confirmed that Age (skewness: 1.53) and NCP (skewness: -1.11) deviate most from normality. Variables identified for potential log transformation were Age, FCVC, NCP, CH2O, FAF, and TUE. However, because Neural Networks and Logistic Regression do not strictly require normally distributed inputs, log transformation was not applied.

### 3. Feature Selection

A correlation heatmap of the numerical features showed no multicollinearity (all pairwise correlations below 0.7), meaning no features were dropped on grounds of redundancy.

LASSO regression (`alpha=0.01`) was used to rank feature importance. The most informative features by coefficient magnitude were:

| Feature | LASSO Coefficient |
|---|---|
| `CAEC` (food between meals) | 0.901 |
| `family_history_with_overweight` | 0.749 |
| `FAVC` (high-caloric food) | -0.566 |
| `Weight` | 0.526 |
| `Age` | 0.344 |
| `CALC` (alcohol) | -0.340 |

All 16 features were retained for modelling, with LASSO informing interpretation rather than filtering.

### 4. Train-Test Split

The dataset was split 70/30 into training and test sets using stratified sampling to preserve class proportions.

```
Training set:  1,477 samples × 16 features
Test set:        634 samples × 16 features
```

### 5. Model Training

Two model families were trained and evaluated:

**One-vs-Rest Logistic Regression (Base)**

A `OneVsRestClassifier` wrapping `LogisticRegression(solver='lbfgs')` was trained as a baseline. This decomposes the 7-class problem into 7 binary classifiers, each learning to distinguish one class from all others. The base model achieved **75% accuracy**.

**Sequential Neural Network (Base)**

A 3-layer Keras `Sequential` model with Dropout regularisation:

```
Input (16) → Dense(128, ReLU) → Dropout(0.25)
           → Dense(64, ReLU)  → Dropout(0.20)
           → Dense(7, Softmax)
```

Compiled with Adam and `sparse_categorical_crossentropy`. Early stopping (`patience=5`) was applied on validation loss. The base NN achieved **73% accuracy**.

### 6. Hyperparameter Tuning

**Tuned OVR (GridSearchCV)**

A 5-fold cross-validated grid search was conducted over regularisation strength `C` ∈ {0.01, 0.1, 1, 10, 100}. The optimal configuration was `C=100` with L2 penalty, which improved accuracy to **77%**.

**Tuned Neural Network (Keras Tuner — RandomSearch)**

Hyperparameter search was performed over: number of hidden units (32–128), dropout rate (0.2–0.4), and learning rate ({1e-2, 1e-3, 3e-4}). After 8 trials, the best model architecture was:

```
Input (16) → Dense(64, ReLU) → Dropout
           → Dense(32, ReLU) → Dropout
           → Dense(7, Softmax)
```

Total parameters: 4,631. The tuned NN achieved **92.6% accuracy** on the test set.

### 7. Model Explainability

**SHAP (Kernel Explainer)** was applied to the tuned Neural Network to understand the contribution of each feature to individual predictions across all 7 classes. Beeswarm plots were generated per class (0–6), visualising how each feature pushes predictions toward or away from each obesity level.

Per-class coefficient analysis was also performed on the OVR model by inspecting the `coef_` attribute of each of the 7 internal binary classifiers, providing a transparent linear interpretation of the logistic model's decision boundaries.

---

## Results

### Model Comparison

| Model | Accuracy | Macro F1 |
|---|---|---|
| OVR Logistic Regression (Base) | 75% | 0.74 |
| OVR Logistic Regression (Tuned) | 77% | 0.77 |
| Sequential Neural Network (Base) | 73% | ~0.71 |
| **Sequential Neural Network (Tuned)** | **92.6%** | **~0.92** |

### Tuned Neural Network — Per-Class Performance

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Insufficient Weight | 0.92 | 0.99 | 0.95 |
| Normal Weight | 0.85 | 0.84 | 0.84 |
| Overweight Level I | 0.99 | 0.91 | 0.95 |
| Overweight Level II | 0.98 | 0.99 | 0.98 |
| Obesity Type I | 1.00 | 0.99 | 0.99 |
| Obesity Type II | 0.88 | 0.80 | 0.84 |
| Obesity Type III | 0.86 | 0.97 | 0.91 |
| **Macro Average** | **0.92** | **0.93** | **0.92** |

### Tuned OVR — Per-Class Performance

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Insufficient Weight | 0.99 | 1.00 | 0.99 |
| Normal Weight | 0.70 | 0.60 | 0.65 |
| Overweight Level I | 0.65 | 0.75 | 0.70 |
| Overweight Level II | 0.91 | 0.97 | 0.94 |
| Obesity Type I | 1.00 | 0.99 | 0.99 |
| Obesity Type II | 0.62 | 0.61 | 0.61 |
| Obesity Type III | 0.53 | 0.48 | 0.50 |
| **Macro Average** | **0.77** | **0.77** | **0.77** |

Test loss for the tuned NN: **0.191**

---

## Conclusion

The tuned Sequential Neural Network significantly outperformed the One-vs-Rest Logistic Regression across all metrics, achieving **92.6% accuracy** and a macro F1 of **0.92** compared to **77%** and **0.77** for the tuned OVR model. The neural network performed especially strongly on extreme obesity classes (Obesity Type I: F1 = 0.99), while the most ambiguous boundary classes (Normal Weight, Overweight Level I, Obesity Types II and III) showed the most room for improvement across both models.

The primary trade-off is interpretability: while the OVR model offers transparent per-class coefficients and linear decision boundaries, the Neural Network requires post-hoc explainability tools (SHAP) to understand its predictions. For deployment contexts where interpretability is non-negotiable, the tuned OVR model provides a reasonable and well-understood alternative at 77% accuracy.

The best-performing Neural Network model was serialised and saved using `pickle` for downstream use.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, OVR model, LASSO, evaluation |
| `tensorflow` / `keras` | Sequential Neural Network |
| `keras_tuner` | Neural Network hyperparameter search |
| `shap` | Model explainability (SHAP KernelExplainer) |
| `matplotlib`, `seaborn` | Visualisation |
| `statsmodels` | Q-Q plots, normality assessment |
| `scipy` | Skewness and kurtosis computation |
| `pickle` | Model serialisation |

---

## How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn tensorflow keras-tuner shap matplotlib seaborn statsmodels scipy
   ```
3. Download the [Obesity Levels Dataset](https://www.kaggle.com/datasets/fabiogreis/obesitydataset) and update the file path in the notebook
4. Run `multiclass_obesity_notebook.ipynb` from top to bottom

To load the saved model:

```python
import pickle

with open('finalized_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

predictions = loaded_model.predict(X_test)
```
