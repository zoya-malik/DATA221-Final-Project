# DATA221 Final Project — Credit Card Default Prediction

**University of Calgary | DATA 221: Introduction to Data Science | Winter 2026**

**Team:** Fazel Rabbi, Zoya Malik, Siyi Ye

---

## Overview

This project looks at predicting whether a credit card applicant is likely to default on their payments. We used a publicly available Kaggle dataset and trained four different machine learning models — Logistic Regression, Decision Tree, K-Nearest Neighbours (KNN), and a Neural Network to see how they compare on this kind of imbalanced classification problem.

The dataset had a significant class imbalance (about 88% non-defaulters vs. 12% defaulters), so we put extra focus on recall and ROC-AUC instead of just accuracy. We also designed a composite scoring formula to rank the models in a way that reflects the real cost of missing a high-risk applicant.

---

## Dataset

**Source:** [Credit Card Approval Prediction — Kaggle](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

The dataset consists of two files that were merged on a shared client ID:

- `application_record.csv` — demographic and financial information per applicant (income, employment, education, etc.)
- `credit_record.csv` — monthly credit status history per client

After merging and cleaning, the final dataset had **36,457 records** and **17 features**. The target label (`default`) was derived from the credit record: any client with a status of 2 or worse (60+ days overdue) was labeled as a defaulter.

---

## Project Structure

```
DATA221-Final-Project/
├── data/
│   └── cleaned_data.csv          # Preprocessed dataset (raw CSVs excluded)
├── notebooks/
│   ├── data_preprocessing.ipynb  # Merging, cleaning, feature engineering
│   ├── logistic_regression.ipynb # Logistic Regression model (Zoya)
│   ├── decision_tree.py          # Decision Tree with depth tuning (Siyi)
│   ├── Neural-Network-Credit-Card-Approval.py  # Neural Network (Fazel)
│   └── KNN-Credit-Card-Approval.py             # KNN model (Fazel)
├── results/
│   └── *.png                     # All figures generated from notebooks
├── report/
│   ├── DATA221_Final_Report.tex  # LaTeX source (IEEE two-column format)
│   └── figures/                  # Figures used in the report
├── proposal/
│   └── Project_Proposal_Data_221.pdf
├── presentation/
│   └── DATA221Presentation_final.pptx
├── .gitignore
└── README.md
```

---

## Models

We trained all four models on an 80/20 train-test split using the same cleaned dataset.

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC | Composite Score |
|---------------------|----------|-----------|--------|-------|---------|-----------------|
| Logistic Regression | 58.3%    | 13.5%     | 49.2%  | 21.2% | 61.9%   | 0.421           |
| Decision Tree       | 67.1%    | 15.2%     | 39.2%  | 21.8% | 57.9%   | 0.398           |
| Neural Network      | 85.0%    | 33.1%     | 20.5%  | 25.3% | 62.8%   | 0.397           |
| **KNN (K=1)**       | **85.5%**| **38.5%** |**37.4%**|**37.9%**|**62.3%**| **0.483**    |

**KNN ranked first** based on our composite scoring formula, which weighted recall and ROC-AUC more heavily to reflect the cost of missing a defaulter.

### Composite Score Formula

```
S = 0.35 × Recall + 0.30 × ROC-AUC + 0.20 × F1 + 0.10 × Precision + 0.05 × Accuracy
```

---

## How to Run

### Requirements

```
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

### Steps

1. Download `application_record.csv` and `credit_record.csv` from Kaggle and place them in the `data/` folder.
2. Run `notebooks/data_preprocessing.ipynb` to generate `cleaned_data.csv` (OR download cleaned_data.csv from this repo and skip step 1)
3. Run any of the model notebooks/scripts from the `notebooks/` folder.

Each model script is self-contained, just make sure `cleaned_data.csv` exists in `data/` before running.

---

## Key Findings

- KNN with K=1 had the best composite score but is sensitive to noise, results may vary with different random seeds.
- Logistic Regression was a strong baseline, ranking second despite being the simplest model.
- The Neural Network had the highest accuracy but struggled with recall on the minority class, which hurt its composite score.
- Class imbalance was the biggest challenge throughout. We handled it using class weights and threshold tuning rather than resampling.

---

## Limitations

- We only used a single 80/20 split, so results aren't cross-validated.
- The dataset only includes demographic and financial history, no behavioral or transactional data, which is usually more predictive of default.
- KNN with K=1 may overfit and not generalize well to truly new data.

---

## Repository

[https://github.com/zoya-malik/DATA221-Final-Project](https://github.com/zoya-malik/DATA221-Final-Project)

---

## Authors

- **Zoya Malik** — data preprocessing, Logistic Regression model
- **Fazel Rabbi** — Neural Network and KNN models
- **Siyi Ye** — Decision Tree model
