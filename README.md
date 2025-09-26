# Flipkart Customer Service Satisfaction Prediction  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![ML](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression%2C%20RandomForest-green) ![EDA](https://img.shields.io/badge/EDA-Seaborn%2C%20Matplotlib-orange) ![Deployment](https://img.shields.io/badge/Deployment-Streamlit%2C%20Joblib-purple)  

> **Goal:** Predict Flipkart customer satisfaction (CSAT) using ML models to proactively identify dissatisfied customers and enhance service quality.  

---

## 📑 Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Business Problem](#business-problem)  
4. [End-to-End Workflow](#end-to-end-workflow)  
   - [1. Project Setup](#1-project-setup)  
   - [2. Data Cleaning & Feature Engineering](#2-data-cleaning--feature-engineering)  
   - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)  
   - [4. Hypothesis Testing](#4-hypothesis-testing)  
   - [5. Modeling & Hyperparameter Tuning](#5-modeling--hyperparameter-tuning)  
   - [6. Evaluation](#6-evaluation)  
   - [7. Deployment](#7-deployment)  
5. [Key Results](#key-results)  
6. [Final Conclusions](#final-conclusions)  
7. [Future Scope](#future-scope)  

---

##  Project Overview  
Flipkart relies on customer support to resolve issues, but **low CSAT scores** often signal poor service experiences, risk of churn, and reduced loyalty.  
This project applied **ML + statistical testing** to classify customer satisfaction from support interaction data.  

---

##  Dataset
- **Total records**: ~85,907 support interactions  
- **Features**: 20 structured attributes including  
  - `channel_name`, `category`, `sub_category`, `customer_remarks`, `item_price`, `order_date_time`, `connected_handling_time`, `agent_name`, `tenure_bucket`, `agent_shift`  
- **Target**: `csat_score` (satisfied / dissatisfied)  

---

##  Business Problem
- How can Flipkart **predict dissatisfied customers** before survey results arrive?  
- Which **interaction categories** (returns, order-related, etc.) drive dissatisfaction?  
- Can these insights guide **agent training & process optimization**?  

---

##  End-to-End Workflow

### 1. Project Setup  
- Virtual environment (conda / venv)  
- Libraries: pandas, numpy, sklearn, seaborn, matplotlib, joblib, streamlit  

---

### 2. Data Cleaning & Feature Engineering  
- Filled missing `customer_remarks` with blanks  
- Created flags: `has_order_date`, `is_connected_call`  
- Handled outliers in `item_price`  
- Converted timestamps → numeric (Unix epoch)  
- One-hot encoded categorical variables  

---

### 3. Exploratory Data Analysis  
We performed detailed EDA to uncover **patterns in customer interactions** and their relationship with CSAT scores.
Visualizations revealed: 
- CSAT distribution is **skewed toward satisfied customers** (imbalance).  
- Dissatisfaction is highest in **Returns-related interactions**.  
- **Inbound calls** report lower CSAT compared to other channels.  
- **Item price** is highly skewed → outliers capped.  
- **Agent tenure** correlates with service quality.

1. **Distribution of Customer Service Channel**  
   - Shows share of Inbound, Outbound, Chat, Email.  <img width="403" height="271" alt="Distribution of Customer Service Channel" src="https://github.com/user-attachments/assets/e553c68e-7f9d-4375-9d51-67e2c734407d" /> 

2. **Item Price Boxplot**  
   - Highlights extreme outliers in item prices; capped for modeling.  <img width="412" height="160" alt="Item Price Boxplot" src="https://github.com/user-attachments/assets/e42a1435-130c-40c4-8ede-4af544916f44" />
 

3. **Item Price Histogram**  
   - Confirms right-skewed distribution of prices.  <img width="481" height="314" alt="Item Price Histogram" src="https://github.com/user-attachments/assets/e6e863dd-3f4e-4587-be2a-78dddf604e05" />

4. **Agent Tenure Buckets (Barplot)**  
   - Shows CSAT distribution across new vs experienced agents.  <img width="616" height="468" alt="Agent Tenure Buckets" src="https://github.com/user-attachments/assets/91181f32-9d35-40cb-8034-d56d7108a45b" />

5. **Average CSAT Score by Interaction Category**  
   - Returns-related cases show the lowest CSAT.  <img width="968" height="566" alt="CSAT by Category" src="https://github.com/user-attachments/assets/445a783b-1861-41cf-bf01-2a9a667aa048" />
   
6. **Average CSAT Score by Channel Name**  
   - Inbound channels trend toward dissatisfaction.  <img width="981" height="510" alt="CSAT by Channel" src="https://github.com/user-attachments/assets/2bbd53eb-9d89-4969-9cb6-b3e56456bc36" />

   
7. **Average CSAT by Interaction Category × Channel**  
   - Heatmap reveals combinations with highest dissatisfaction (e.g., Returns + Inbound). <img width="988" height="567" alt="CSAT by Category and Channel" src="https://github.com/user-attachments/assets/1f5ee3d9-9b37-4b39-a286-fecbc158f522" />

---

### 4. Hypothesis Testing  
- **ANOVA**: CSAT differs significantly across communication channels  
- **ANOVA**: CSAT differs across issue categories  
- **t-test**: CSAT differs with vs without order date provided  

---

### 5. Modeling & Hyperparameter Tuning  
- **Models tried**: Logistic Regression (baseline), Random Forest (nonlinear)  
- Used **GridSearchCV** for hyperparameter tuning (depth, estimators, features)  
- Scoring prioritized **recall** for dissatisfied customers (false negatives are costly in business).  

---

### 6. Evaluation  
Classification Report (Random Forest best):  
              precision    recall  f1-score   support
Dissatisfied    0.47        0.02    0.05       3014
Satisfied       0.83        0.99    0.90      14168
accuracy                                0.82  17182
macro avg       0.65        0.51    0.47      17182
weighted avg    0.77        0.82    0.75      17182


- **Accuracy**: ~82%  
- **High recall** for satisfied class  
- Dissatisfied recall low → flagged for future improvements  

---

### 7. Deployment  
- Saved best model as `best_model.pkl` (joblib)  
- Built a **Streamlit web app**:  
  - Sidebar: user input fields  
  - Batch prediction via CSV upload  
  - Outputs predictions in real-time  

---

##  Key Results  
- **Best Model**: Random Forest, accuracy = 82%  
- **Top predictors**: `item_price`, `category_Returns`, `has_order_date`, `tenure_bucket`  
- **Business Insight**: Returns and order-related issues dominate dissatisfaction  

---

## Final Conclusions  
- Flipkart’s CSAT strongly depends on **channel, category, and order info availability**  
- Random Forest provided good baseline performance but needs improvement in detecting dissatisfied customers  
- Model pipeline + Streamlit app demonstrate **end-to-end ML deployment skills**  

---

##  Future Scope  
- Incorporate **text analytics** of customer remarks (TF-IDF, BERT)  
- Use **XGBoost / Transformers** for better minority class recall  
- Add **SHAP explainability** for trust in predictions  
- Deploy at scale with **batch + streaming pipelines**  

---

📎 **Author:** [Uttam Singh Chaudhary](https://www.linkedin.com/in/uttam-singh-chaudhary-98408214b)  

