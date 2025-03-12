# Credit Risk Analysis Model

This project is an advanced model for credit risk analysis, developed with **Streamlit**, **scikit-learn**, **XGBoost**, **SHAP**, **LIME**, and other data science libraries. It enables data loading, feature engineering, predictive modeling, and model explainability, including interactive visualizations and scenario simulations.

## 🚀 Features

- **📂 Data Upload & Exploratory Data Analysis**
  - Supports CSV file uploads with automatic preprocessing.
  - Descriptive statistics and interactive visualizations of financial variables.

- **📊 Predictive Modeling & Evaluation**
  - Implementation of supervised models for default prediction.
  - Supported algorithms: XGBoost, RandomForest, Gradient Boosting, AdaBoost, SVM, MLP, Naïve Bayes, and Logistic Regression.
  - Data balancing techniques such as SMOTE.
  - Evaluation metrics: Accuracy, AUC-ROC, Precision, Recall, F1-score, Confusion Matrix, and Lift Curve.
  - Dynamic hyperparameter tuning for model optimization.

- **🔎 Explainability & Interpretation**
  - Model interpretation using SHAP, LIME, and variable impact analysis.
  - Visualizations such as SHAP Summary Plot, Dependence Plot, Waterfall Plot, and Force Plot.
  - Helps understand model decisions for each prediction.

- **⚙️ Simulations & Sensitivity Analysis**
  - "What-if" scenarios to evaluate the impact of key variables.
  - Robustness testing with varying customer characteristics.

- **📄 Report Generation**
  - Export comprehensive reports in **HTML** with visualizations and statistical analysis.
  - Model comparison with a summary of performance metrics.

## 🛠️ Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/your-username/credit-risk-model.git
   cd credit-risk-model
   ```

2. **Create a virtual environment (optional but recommended)**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

## ▶️ Usage

To run the model and visualize the results:
```sh
streamlit run app.py
```

Access the Streamlit-provided link in your browser.

### 📌 Project Structure
```
📦 credit-risk-model
 ┣ 📜 README.md             # Project documentation
 ┣ 📜 app.py                # Main dashboard script containing all project logic
 ┣ 📜 requirements.txt      # Project dependencies
 ┣ 📜 bankrisk_client_data.csv  # Sample dataset for testing
```

## 🤝 Contributing

Contributions are welcome! To contribute:
1. **Fork** this repository.
2. Create a new branch: `git checkout -b my-feature`.
3. Make your changes and commit: `git commit -m 'My new feature'`.
4. Push to GitHub: `git push origin my-feature`.
5. Open a Pull Request.

## 📜 License

This project is distributed under the MIT license. See the LICENSE file for details.

---

📌 *An advanced model for credit risk prediction using AI and Machine Learning.*

