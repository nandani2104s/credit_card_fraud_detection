# Credit Card Fraud Detection 💳🛡️

## 📌 Project Overview
This project focuses on identifying fraudulent credit card transactions using Machine Learning. Given the highly imbalanced nature of the dataset (where most transactions are genuine), we implement specialized techniques like **Feature Scaling**, **SMOTE (Oversampling)**, and **Under-sampling** to ensure the model accurately detects rare fraud cases.

## 📊 Dataset Information
The dataset used is the popular Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset.
* **Transactions:** 284,807
* **Fraudulent Cases:** 492 (0.17%)
* **Features:** 28 PCA-transformed features ($V1$ to $V28$), 'Time', and 'Amount'.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE), Matplotlib, Seaborn.
* **Environment:** VS Code / Jupyter Notebook

## 🚀 Key Features & Workflow
1. **Exploratory Data Analysis (EDA):** Analyzing class distributions and transaction patterns.
2. **Data Preprocessing:** - Scaling 'Time' and 'Amount' using `RobustScaler`.
   - Handling the extreme class imbalance.
3. **Model Implementation:**
   - Logistic Regression
   - Random Forest Classifier (or XGBoost)
4. **Evaluation Metrics:** - Focused on **Precision-Recall Curves**, **F1-Score**, and **Confusion Matrices** (since Accuracy is misleading for imbalanced data).

## 💻 How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/nandani2104s/credit_card_fraud_detection.git](https://github.com/nandani2104s/credit_card_fraud_detection.git)
