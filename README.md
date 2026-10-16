# Customer Churn Prediction

**Predicting customer churn risk using Machine Learning and Streamlit**

## Project Overview

Customer churn is a major challenge for subscription-based businesses. When customers leave, businesses can lose recurring revenue and incur additional customer acquisition costs.

This project uses machine learning to estimate customer churn probability and identify customers who may need retention attention.

The model was trained using the **IBM Telco Customer Churn dataset** and deployed as an interactive web application using Streamlit.

## Business Problem

How can a business identify customers who are more likely to leave so that its retention team can prioritize proactive engagement?

This project demonstrates how historical customer information can be used to estimate churn risk and support data-driven retention decisions.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model:** Logistic Regression
- **Preprocessing:** StandardScaler, OneHotEncoder, ColumnTransformer
- **Web Application:** Streamlit
- **Model Serialization:** Joblib
- **Development:** Google Colab, VS Code
- **Version Control:** GitHub

## Machine Learning Workflow

1. Loaded and explored the customer churn dataset.
2. Investigated missing values, data types, and class distribution.
3. Converted `TotalCharges` into a numeric format and handled 11 blank values.
4. Separated input features from the target variable.
5. Split the data into training, validation, and test datasets using stratified sampling.
6. Scaled numerical features and one-hot encoded categorical features.
7. Trained a Logistic Regression classifier.
8. Compared classification thresholds using the validation dataset.
9. Selected a threshold of **0.30** to prioritize churn detection.
10. Evaluated the final model on a separate test dataset.
11. Saved the fitted preprocessing pipeline, model, feature lists, and threshold in a Joblib artifact.
12. Built an interactive Streamlit application for customer-level predictions.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

- Total records: 7,043
- Input features: 19
- Target variable: `Churn`
- Original target distribution:
  - Customers who did not churn: 5,174
  - Customers who churned: 1,869

The `customerID` column was excluded from model training because it is an identifier rather than a useful customer behavior feature.

## Model Evaluation

The final evaluation used a separate test dataset that was not used for training or threshold selection.

| Metric | Test Result |
|---|---:|
| Accuracy | 76.44% |
| Precision | 54.04% |
| Recall | 75.13% |
| F1-score | 62.86% |
| ROC-AUC | 0.8465 |

### Why was the threshold set to 0.30?

The default classification threshold of 0.50 can miss customers who are at risk of churning.

A threshold of 0.30 was selected using validation data to prioritize recall while considering the F1-score.

At this threshold, the final test confusion matrix was:

| Actual / Predicted | No Churn | Churn |
|---|---:|---:|
| No Churn | 796 | 239 |
| Churn | 93 | 281 |

The model identified 281 of the 374 actual churners in the test set.

The lower threshold also produced false positives. Businesses should consider retention campaign costs, customer lifetime value, and the cost of missed churn before choosing a production threshold.

## Streamlit Application

The web application allows users to:

- Enter customer details through an interactive form.
- Generate an estimated churn probability.
- Classify the customer as higher or lower risk using the selected threshold.
- Explore how customer characteristics translate into a model prediction.

**Important:** The prediction is an estimate, not a guarantee that a customer will or will not churn.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows CMD:

```bat
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The requirements file pins scikit-learn to version 1.6.1 to match the saved model's training environment.

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, typically `http://localhost:8501`.

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── customer_churn_model.joblib
├── requirements.txt
└── README.md
```

## Limitations and Future Improvements

- Compare Logistic Regression with tree-based models such as Random Forest and Gradient Boosting.
- Evaluate probability calibration and monitor performance after deployment.
- Estimate the financial impact of false positives and false negatives.
- Incorporate retention campaign costs and customer lifetime value into threshold selection.
- Monitor changes in customer behavior and model performance over time.
- Add batch prediction and downloadable prediction reports.

## Key Learning Outcomes

This project demonstrates an end-to-end machine learning workflow,
