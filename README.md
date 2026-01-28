# 📊 YouTube Ad Revenue Prediction & Analysis

## 📌 Project Overview

This project focuses on analyzing **YouTube video performance data** and building a **machine learning regression system** to predict **ad revenue (USD)**.
The solution follows a complete **end-to-end data science workflow** including data cleaning, exploratory data analysis, feature engineering, model comparison, and deployment using **Streamlit Cloud**.

---

## 🎯 Problem Statement

YouTube creators and digital marketers want to understand:

* What factors influence ad revenue?
* Can ad revenue be predicted before publishing content?
* Which metrics matter most for monetization?

This project answers these questions using **regression-based predictive modeling**.

---

## 🗂 Dataset Description

The dataset contains video-level performance data with the following features:

| Column Name            | Description             |
| ---------------------- | ----------------------- |
| `video_id`             | Unique video identifier |
| `date`                 | Upload date             |
| `views`                | Total views             |
| `likes`                | Total likes             |
| `comments`             | Total comments          |
| `watch_time_minutes`   | Total watch time        |
| `video_length_minutes` | Video duration          |
| `subscribers`          | Channel subscribers     |
| `category`             | Video category          |
| `device`               | Device type             |
| `country`              | Viewer country          |
| `ad_revenue_usd`       | **Target variable**     |

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights from EDA:

* **Watch time and views** have the strongest correlation with ad revenue
* Engagement metrics (likes, comments) have moderate impact
* Video length alone does not guarantee higher revenue
* Revenue varies by **country** and **device type**

Visualizations used:

* Correlation heatmap
* Revenue distribution
* Category-wise revenue comparison

---

## 🧹 Data Preprocessing

Steps performed:

* Missing value handling (median / most frequent)
* Duplicate removal
* Text standardization (lowercasing categorical values)
* Outlier handling (IQR method)
* Categorical encoding using **OneHotEncoder**
* Feature scaling using **StandardScaler**

---

## 🛠 Feature Engineering

* Selected relevant numerical and categorical features
* Ensured consistent feature alignment using **scikit-learn pipelines**
* Prevented data leakage by fitting transformations only on training data

---

## 🤖 Model Building & Comparison

Five regression models were trained and evaluated:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Random Forest Regressor
* Gradient Boosting Regressor

### 📈 Evaluation Metrics

* **R² Score**
* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**

The **best-performing model** was saved as a complete preprocessing + model pipeline.

---

## 📊 Model Evaluation Summary

* Tree-based models performed better for non-linear relationships
* Linear models were more interpretable and stable
* Final model was selected based on **balanced performance across all metrics**

---

## 🌐 Streamlit Web Application

An interactive Streamlit app was built to:

* Explore dataset insights
* Visualize correlations
* Predict ad revenue based on user inputs
* Display business recommendations

### 🔗 Live App

👉 *Deployed on Streamlit Cloud*
https://youtube-ad-revenue-prediction-vinay123.streamlit.app/

---

## 🚀 Deployment

The application was deployed using **Streamlit Cloud** with:

* Pre-trained model pipeline (`.pkl`)
* `requirements.txt` for dependency management
* GitHub-based CI/CD deployment

---

## 🧰 Tech Stack

* **Python**
* **Pandas & NumPy**
* **Matplotlib & Seaborn**
* **Scikit-learn**
* **Streamlit**
* **GitHub**
* **Streamlit Cloud**

---




## 📈 Future Improvements

* Add time-based trend analysis
* Introduce engagement rate features
* Add confidence intervals to predictions
* Integrate external monetization factors

---

## 👤 Author

**Vinay Mishra**
📧 vm4253979@gmail.com


---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it helps!


