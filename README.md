# 💰 Adult Income Prediction
 
A machine learning classification project that predicts whether an individual's annual income exceeds $50K using the **UCI Adult Census Income dataset**. The project follows the CRISP-DM methodology, covering the full data science pipeline from data cleaning to model evaluation.
 
---
 
## 📌 Project Overview
 
This project builds a **Logistic Regression classifier** to predict adult income levels based on demographic and employment features. To address the class imbalance in the dataset, **SMOTE (Synthetic Minority Oversampling Technique)** was applied, followed by hyperparameter tuning using **GridSearchCV** with regularization strategies (L1, L2, ElasticNet).
 
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
1. Data Loading & Understanding
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Baseline Model
5. Handling Class Imbalance (SMOTE)
6. Hyperparameter Tuning (GridSearchCV)
7. Final Model Evaluation
8. Feature Importance Analysis
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
 
### Baseline Logistic Regression
Initial model trained on the imbalanced dataset.
 
### After SMOTE + GridSearchCV Tuning
 
| Metric | <=50K | >50K |
|--------|-------|------|
| Recall | 80% | 84% |
| Precision | 94% | 57% |
| **Overall Accuracy** | **81%** | |
 
- **No overfitting detected** — training and test performance were consistent.
- **Best parameters** found via GridSearchCV using `recall_macro` as the scoring metric.
- Explored: L1, L2, ElasticNet, and no-penalty configurations.
---
 
## 🏆 Top Feature Importances (Permutation Importance)
 
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
| 9-10 | `race_Black` | Demographic feature in census data |
 
---
 
## 🛠️ Tech Stack
 
| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Data visualization |
| `scikit-learn` | Preprocessing, modeling, evaluation |
| `imbalanced-learn` | SMOTE for class imbalance |
| Google Colab | Development environment |
 
---
 
## 📊 Key Findings
 
- **Marital status** is the most impactful predictor — married individuals (civ-spouse) have ~45% probability of earning >$50K.
- **Education level** directly correlates with income: individuals at levels 15–16 are most likely to earn high incomes.
- **SMOTE** significantly improved the model's ability to detect high-income individuals (recall improved for the minority class).
---
 
## 👩‍💻 Author
 
**Insaf AlRumi**  
Computer Systems Engineer | Data Science & ML Enthusiast  
Axsos Academy — Data Science & Machine Learning Bootcamp
