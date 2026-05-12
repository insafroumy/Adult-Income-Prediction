# 💰 Adult Income Prediction
 
A machine learning classification project that predicts whether an individual's annual income exceeds $50K using the **UCI Adult Census Income dataset**. The project follows the CRISP-DM methodology, covering the full data science pipeline from data cleaning to model evaluation, feature engineering, and feature selection.
 
---
 
## 📌 Project Overview
 
This project builds a **Logistic Regression classifier** to predict adult income levels based on demographic and employment features. To address the class imbalance in the dataset, **SMOTE (Synthetic Minority Oversampling Technique)** was applied, followed by hyperparameter tuning using **GridSearchCV** with regularization strategies (L1, L2, ElasticNet). In Part 2, the project extends into **feature engineering** (PCA + KMeans Clustering) and **feature selection** (embedded method using Logistic Regression coefficients) to evaluate their impact on model performance.
 
---
 
## 📁 Dataset
 
- **Source:** UCI Adult Census Income Dataset (`adult.csv`)
- **Size:** 48,842 rows × 15 features (after removing 52 duplicate rows)
- **Target Variable:** `income` — Binary: `<=50K` or `>50K`
- **Feature Types:** 6 numeric, 9 categorical
> **Note:** Missing values were encoded as `?` in the original dataset and replaced with `NaN` for proper handling (total of 6,456 missing values).
 
---
 
## 🔄 Project Workflow (CRISP-DM)
 
```
Part 1:
1. Data Loading & Understanding
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Baseline Model
5. Handling Class Imbalance (SMOTE)
6. Hyperparameter Tuning (GridSearchCV)
7. Final Model Evaluation
8. Feature Importance Analysis (Permutation Importance)
 
Part 2:
9.  KMeans Clustering (Feature Engineering)
10. PCA — Principal Component Analysis (Feature Engineering)
11. Feature Selection — Embedded Method (Logistic Regression Coefficients)
12. Model Comparison Across All Configurations
```
 
---
 
## 🔍 Exploratory Data Analysis
 
Key insights from the visualizations:
 
- **Work Class:** Most individuals work in the **Private sector**.
- **Occupation:** `Prof-specialty`, `Craft-repair`, and `Exec-managerial` are the most common.
- **Marital Status:** The majority are either **Married (civ-spouse)** or **Never-married**.
---
 
## ⚙️ Preprocessing Pipeline
 
| Step | Feature Type | Transformations Applied |
|------|-------------|------------------------|
| Imputation | Categorical | `SimpleImputer` (most_frequent) |
| Encoding | Categorical | `OneHotEncoder` (drop='first') |
| Scaling | Numeric | `StandardScaler` |
 
- **Dropped columns:** `education` (redundant with `educational-num`), `fnlwgt` (census weight, not predictive)
- Pipeline built using `ColumnTransformer` with `verbose_feature_names_out=False`
---
 
## 🤖 Models & Results
 
### Part 1 — Baseline & Tuned Logistic Regression
 
| Model | Accuracy | Recall <=50K | Recall >50K | Precision >50K |
|-------|----------|--------------|-------------|----------------|
| Baseline LR | — | — | — | — |
| SMOTE + GridSearchCV (Best) | **81%** | 80% | 84% | 57% |
 
- **No overfitting detected** — training and test performance were consistent.
- **Best parameters** found via GridSearchCV using `recall_macro` as the scoring metric.
- Explored: L1, L2, ElasticNet, and no-penalty configurations.
---
 
### Part 2 — Feature Engineering & Selection
 
#### 🔵 KMeans Clustering
 
- Applied **KMeans** on the scaled training data to group similar individuals.
- Used the **Elbow Method** and **Silhouette Score** to select the optimal number of clusters.
- **5 clusters** were selected as the best choice based on low inertia and high silhouette score.
- Cluster labels were added as an additional feature alongside the original features.
- **PCA (3 components)** was applied afterward to visualize the clusters in 3D space.
#### 🔵 PCA — Principal Component Analysis
 
- Applied **PCA with 3 principal components** on the SMOTE-resampled training data.
- Used `pca.transform()` on test data (fitted only on training data — no data leakage).
- Trained a Logistic Regression model using only the 3 PCs.
- **Result:** Performance using only PCs was worse than using the original features → original features were kept.
#### 🔵 Feature Selection — Embedded Method
 
- Used **Logistic Regression coefficients** via `SelectFromModel` (default threshold = mean of absolute coefficients).
- Selected only features whose coefficient magnitude exceeded the threshold.
- Trained a final Logistic Regression model on the selected features.
#### 📊 Model Comparison Summary
 
| Model | Accuracy | Recall <=50K | Recall >50K | Precision >50K |
|-------|----------|--------------|-------------|----------------|
| Best Model (Part 1) | **81%** | 80% | 84% | 57% |
| LR with PCA only | < 81% | — | — | — |
| LR after Feature Selection | 79% | 77% | 85% | 54% |
 
**Key observations:**
- Feature selection slightly **reduced overall accuracy** (−2%) and precision for `>50K` (−3%).
- However, it **improved recall for `>50K`** slightly (+1%), meaning fewer high-income individuals were missed.
- False positive errors increased by 1% and false negative errors increased by 3%.
- The original tuned model from Part 1 remains the best-performing configuration.
---
 
## 🏆 Top Feature Importances (Permutation Importance — Part 1 Model)
 
| Rank | Feature | Insight |
|------|---------|---------|
| 1 | `marital-status_Married-civ-spouse` | Strongest predictor of high income |
| 2 | `educational-num` | Higher education → higher income likelihood |
| 3 | `capital-gain` | Significant financial indicator |
| 4 | `workclass_Self-emp-not-inc` | Self-employment type matters |
| 5 | `relationship_Not-in-family` | Household role influences income |
| 6 | `occupation_Exec-managerial` | Executive roles linked to higher pay |
| 7 | `occupation_Farming-fishing` | Notable occupational predictor |
| 8 | `occupation_Other-service` | Service sector patterns |
| 9–10 | `race_Black` | Demographic feature in census data |
 
---
 
## 🛠️ Tech Stack
 
| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn`, `plotly` | Data visualization |
| `scikit-learn` | Preprocessing, modeling, evaluation, PCA, KMeans, feature selection |
| `imbalanced-learn` | SMOTE for class imbalance |
| Google Colab | Development environment |
 
---
 
## 📊 Key Findings
 
- **Marital status** is the most impactful predictor — married individuals (civ-spouse) have ~45% probability of earning >$50K.
- **Education level** directly correlates with income: individuals at levels 15–16 are most likely to earn high incomes.
- **SMOTE** significantly improved the model's ability to detect high-income individuals (recall improved for the minority class).
- **PCA alone** (3 components) was insufficient to match the performance of the full original feature set.
- **Feature selection** via embedded method slightly reduced accuracy but offered a more interpretable, compact model.
- The **best overall model remains the SMOTE + GridSearchCV Logistic Regression** from Part 1 with 81% accuracy.
---
 
## 👩‍💻 Author
 
**Insaf AlRumi**
Computer Systems Engineer | Data Science & ML Enthusiast
Axsos Academy — Data Science & Machine Learning Bootcamp
