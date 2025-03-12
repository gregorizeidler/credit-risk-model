# Credit Risk Analysis Model

This project is an advanced **Data Science** model for credit risk analysis, developed with **Streamlit**, **scikit-learn**, **XGBoost**, **SHAP**, **LIME**, and other machine learning libraries. It enables exploratory data analysis, feature engineering, predictive modeling, performance evaluation, and model interpretability, including interactive visualizations and scenario simulations.

## 📊 Data Science Features

- **📂 Exploratory Data Analysis (EDA)**
  - Supports CSV file uploads with automated preprocessing.
  - Descriptive statistics and graphical analysis of financial variables.
  - Detection and treatment of outliers using percentile-based techniques.

- **🤖 Predictive Modeling & Evaluation**
  - Development of supervised models for default prediction.
  - Supported algorithms:
    - `XGBoost`
    - `RandomForest`
    - `Gradient Boosting`
    - `AdaBoost`
    - `SVM`
    - `MLP`
    - `Naïve Bayes`
    - `Logistic Regression`
  - Class balancing with `SMOTE` to mitigate bias in imbalanced datasets.
  - Model performance evaluation using:
    - `Accuracy`
    - `AUC-ROC`
    - `Precision`
    - `Recall`
    - `F1-score`
    - `Confusion Matrix`
    - `Lift Curve`
  - Hyperparameter optimization using `GridSearchCV` and `RandomizedSearchCV`.

- **📈 Feature Engineering & Variable Selection**
  - Creation of custom financial indicators such as:
    - `credit_utilization`: Ratio of average bill value to credit limit.
    - `financial_stability_score`: Relationship between investments and average expenses.
  - Application of `PCA` for dimensionality reduction and latent pattern identification.
  - Feature selection based on `SHAP` and `Permutation Feature Importance`.

- **🔍 Model Interpretability & Explainability**
  - Local and global explanations using `SHAP` and `LIME`.
  - Detailed visualizations to understand the impact of variables on predictions, including:
    - `SHAP Summary Plot`
    - `Dependence Plot`
    - `Waterfall Plot`
    - `Force Plot`
  - Individual contribution analysis for specific customer predictions.

- **🔬 Simulations & Sensitivity Analysis**
  - "What-if" scenario simulations to assess the impact of key variables on predictions.
  - Partial dependence plots (`PDP/ICE`) for multivariate analysis.
  - Model sensitivity analysis for different customer profiles.

- **📑 Report Generation & Model Comparison**
  - Export interactive reports in **HTML** with visualizations and statistical metrics.
  - Comparison of different models for better decision-making.
  - Cross-validation with `StratifiedKFold` to ensure performance robustness.

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

To run the model and visualize results:
```sh
streamlit run app.py
```
Access the Streamlit-generated link in your browser.

### 📂 Project Structure
```
📦 credit-risk-model
 ┣ 📜 app.py                # Main script containing the entire project logic
 ┣ 📜 requirements.txt      # Project dependencies
 ┣ 📜 README.md             # Project documentation
```

## 🤝 Contributing

Contributions are welcome! To contribute:
1. **Fork** this repository.
2. Create a new branch: `git checkout -b my-feature`.
3. Make your changes and commit: `git commit -m 'My new feature'`.
4. Push to GitHub: `git push origin my-feature`.
5. Open a Pull Request.

## 📜 License

This project is distributed under the MIT license. See the `LICENSE` file for details.

---

📌 *An advanced Data Science solution for credit risk prediction using Machine Learning and Explainable AI (XAI).*
