# Credit Risk Analysis Model

## 📌 Overview

This project is an advanced machine learning-based model for **credit risk analysis**, leveraging **Streamlit**, **scikit-learn**, **XGBoost**, **SHAP**, **LIME**, and other data science libraries. It facilitates **data ingestion**, **feature engineering**, **predictive modeling**, and **model explainability**, featuring interactive visualizations and scenario-based simulations.

## 🚀 Key Features

### 📂 Data Upload & Exploratory Data Analysis (EDA)
- Supports **CSV file uploads** with automated preprocessing.
- Provides **descriptive statistics** and **interactive visualizations** for financial variables.
- Detects and handles **missing values**, **outliers**, and **feature correlations**.

### 💡 Predictive Modeling & Evaluation
- Implements **supervised machine learning models** for credit default prediction.
- Supports multiple algorithms:
  - **XGBoost, RandomForest, Gradient Boosting, AdaBoost, SVM, MLP, Naïve Bayes, and Logistic Regression**.
- Integrates **SMOTE** for handling imbalanced datasets.
- Includes performance evaluation metrics:
  - **Accuracy, AUC-ROC, Precision, Recall, F1-score, Confusion Matrix, and Lift Curve**.
- Offers **hyperparameter tuning** to optimize model performance dynamically.

### 🔍 Explainability & Model Interpretation
- Uses **SHAP** and **LIME** to explain individual and global model decisions.
- Provides intuitive visualizations:
  - **SHAP Summary Plot, Dependence Plot, Waterfall Plot, and Force Plot**.
- Helps **identify key features** influencing predictions.

### ⚙️ Simulations & Sensitivity Analysis
- **"What-if" scenario analysis** to assess variable impact.
- Allows users to **adjust financial parameters** to evaluate prediction robustness.

### 📊 Report Generation
- Generates **detailed reports** in **HTML format** with:
  - **Statistical summaries**, **model comparisons**, and **visual explanations**.

### 🧠 Executive Summary with AI
- Uses **OpenAI GPT-4** to generate an **executive summary** of model insights.
- Provides a **technical report** and **operational report** based on model findings.
- Automatically summarizes **feature importance**, **client approval trends**, and **model performance**.

### 🛡️ Virtual Assistant for Project Insights
- An **AI-powered virtual assistant** to answer project-related questions.
- Uses **model predictions**, **feature importance**, and **historical decisions** to generate insights.
- Helps users analyze **specific clients**, **approval criteria**, and **model strengths**.

## 🛠️ Installation Guide

### Step 1: Clone the Repository
```sh
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model
```

### Step 2: Create a Virtual Environment (Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```sh
pip install -r requirements.txt
```

## ▶️ Running the Application

To launch the interactive dashboard:
```sh
streamlit run app.py
```
Then, open the **Streamlit-provided link** in your browser.

## 🎥 Demonstration Video
To see the model in action, watch the **demonstration video**:
- [📺 Main Project Features](https://drive.google.com/file/d/1JUnzbx6KcSLbteSLVEbOgtbZjiHxFOS8/view?usp=share_link)
- [📺 Executive Summary & Virtual Assistant](https://drive.google.com/file/d/1CYCaKSVsauw9BwV9khUtp1YhgHe9K_Uz/view?usp=sharing)

## 🤝 How to Contribute

We welcome contributions! To contribute:
1. **Fork** this repository.
2. Create a new branch: `git checkout -b my-feature`.
3. Implement your changes and commit: `git commit -m 'Added new feature'`.
4. Push your branch: `git push origin my-feature`.
5. Open a **Pull Request** and describe your changes.

### Possible Contributions:
- Adding **new ML models** or improving feature engineering.
- Enhancing **model interpretability** using new techniques.
- Optimizing **performance and scalability**.
- Improving **UI/UX** of the Streamlit dashboard.

## 📜 License

This project is licensed under the **MIT License**. See the **LICENSE** file for details.

---

📌 *A robust AI-powered model for credit risk prediction and explainability.*
