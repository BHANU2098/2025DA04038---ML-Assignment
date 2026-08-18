# ML Assignment 2 — Classification Models with Streamlit

## a. Problem Statement

Predict whether an individual's annual income exceeds $50,000 based on demographic and employment attributes. This is a binary classification problem using the Adult Income dataset from the UCI Machine Learning Repository.

## b. Dataset Description

| Property         | Details                                                         |
|------------------|-----------------------------------------------------------------|
| **Source**       | UCI Machine Learning Repository — Adult Income Dataset          |
| **URL**          | https://archive.ics.uci.edu/ml/datasets/adult                  |
| **Instances**    | 48,842 (after removing missing values: ~45,222)                 |
| **Features**     | 14 (age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country) |
| **Target**       | `income` — binary: `<=50K` (0) or `>50K` (1)                   |
| **Task**         | Binary Classification                                           |
| **Class balance**| ~76% <=50K, ~24% >50K (imbalanced)                             |

## c. GitHub Repository Link

> **[PASTE YOUR GITHUB REPO LINK HERE]**

Repository contains:
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `README.md` — this file
- `test_data.csv` — test split used for evaluation
- `model/train_models.py` — training script
- `model/*.pkl` — saved model and scaler files

## d. Models Used

### Evaluation Metrics Comparison Table

| ML Model Name        | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression  | 0.8279   | 0.8608 | 0.7246    | 0.4598 | 0.5626 | 0.4806 |
| Decision Tree        | 0.8130   | 0.7542 | 0.6054    | 0.6409 | 0.6227 | 0.4989 |
| kNN                  | 0.8340   | 0.8569 | 0.6711    | 0.6091 | 0.6386 | 0.5322 |
| Naive Bayes          | 0.8081   | 0.8644 | 0.7044    | 0.3495 | 0.4672 | 0.3994 |
| Random Forest        | 0.8594   | 0.9110 | 0.7418    | 0.6378 | 0.6859 | 0.5988 |

### Observations on Model Performance

| ML Model Name        | Observation about model performance |
|----------------------|--------------------------------------|
| Logistic Regression  | Achieves solid accuracy (82.79%) and the second-highest AUC (0.8608), indicating good probability calibration. However, recall for the >50K class is low (0.46), meaning it misses many high-income individuals — a common trait on imbalanced data without threshold tuning. |
| Decision Tree        | Has the highest recall among all models (0.6409), making it more aggressive at identifying >50K earners. However, AUC (0.7542) is the lowest, suggesting the probability estimates are less reliable. Prone to overfitting without pruning. |
| kNN                  | Balanced performance across precision and recall (0.67 / 0.61). Benefits from feature scaling. Slower at prediction time compared to other models since it stores all training points. MCC of 0.5322 is a reasonable result. |
| Naive Bayes          | Despite a high AUC (0.8644), the recall for >50K is very low (0.3495) — the model is overly conservative in predicting the minority class. The Gaussian assumption may not hold well for mixed feature distributions in this dataset. |
| Random Forest        | Best overall performer across all 6 metrics: highest accuracy (85.94%), best AUC (0.9110), best F1 (0.6859), and best MCC (0.5988). The ensemble of decision trees reduces variance and generalises well to unseen data. |
| **Overall Winner**   | **Random Forest** — dominates all other models on every metric, especially AUC (0.911) which reflects strong discriminative power on this imbalanced dataset. |

## Live Streamlit App

> **[[PASTE YOUR STREAMLIT APP LINK HERE]](https://2025da04038---ml-assignment-legnpystmehtwvxivhzh7n.streamlit.app/)**

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models and generate test_data.csv
python model/train_models.py

# 3. Launch Streamlit app
streamlit run app.py
```

## Project Structure

```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- model/
│   │-- train_models.py
│   │-- scaler.pkl
│   │-- Logistic_Regression.pkl
│   │-- Decision_Tree.pkl
│   │-- kNN.pkl
│   │-- Naive_Bayes.pkl
│   │-- Random_Forest.pkl
```
