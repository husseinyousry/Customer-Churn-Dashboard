# Customer-Churn-Dashboard
# 🧠 Customer Churn Prediction Dashboard

A full **Data Science project and Streamlit web app** built to analyze and predict customer churn using real transactional data.  
Developed as part of my final project for Epsilon AI’s Data Science Program.

---

## 🚀 Project Overview
This project identifies customers at risk of churning (leaving) based on their purchase and demographic behavior.  
It integrates **data cleaning, feature engineering, model training**, and an **interactive Streamlit dashboard** for visualization and prediction.

### ✨ Key Features
- **Interactive Dashboard** with three tabs:
  1. 🧍‍♂️ *Customer Insights* – demographics, wealth segment, and behavior.
  2. 💳 *Transaction Insights* – spending, profit, and churn correlation.
  3. 🔮 *Churn Prediction* – upload data and predict churn with ML models.
- **3 Machine Learning Models**
  - Logistic Regression  
  - Random Forest  
  - Gradient Boosting (with hyperparameter tuning)
- **Model Explainability** using SHAP values.
- **Dynamic visualizations** with Plotly for deep exploration.

---

## 🧩 Data Used
Cleaned and merged datasets:
- `customer_data.csv`
- `customer_transactions.csv`
- `customer_summary.csv` *(final processed data used in app)*

Each record includes:
- Customer demographics  
- Transactional behavior  
- Product and profit information  
- Calculated churn label

---

## 🛠️ Tech Stack
| Category | Tools |
|-----------|--------|
| Language | Python |
| Data Manipulation | pandas, numpy |
| Visualization | Plotly, seaborn, matplotlib |
| Modeling | scikit-learn |
| Explainability | SHAP |
| Web Framework | Streamlit |
| Version Control | Git & GitHub |

---

## ⚙️ How to Run Locally

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/customer-churn-dashboard.git
cd customer-churn-dashboard
