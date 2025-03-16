import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import os
import openai

# Configure your OpenAI API key (replace with your actual key or set via environment variables)
openai.api_key = ""

from sklearn.decomposition import PCA
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, roc_curve, 
                             precision_recall_curve, f1_score, precision_score, recall_score, 
                             brier_score_loss)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB  # Bayesian model

# For LIME (optional)
try:
    import lime
    import lime.lime_tabular
except ImportError:
    st.warning("Install 'lime' (pip install lime) to use LIME explanations.")

##############################################
# Function to Generate Report for Download (HTML)
##############################################
def generate_report(result_df, metrics, report_text, fig_list):
    buffer = BytesIO()
    content = "<h1>Credit Risk Analysis Report</h1>"
    content += "<h2>Model Comparison</h2>" + result_df.to_html()
    content += "<h2>Additional Metrics</h2><ul>"
    for k, v in metrics.items():
        content += f"<li>{k}: {v:.3f}</li>"
    content += "</ul>" + report_text
    # Save figures to report as images (converted to base64)
    for idx, fig in enumerate(fig_list):
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches="tight")
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        content += f"<h3>Figure {idx+1}</h3><img src='data:image/png;base64,{img_str}' />"
    buffer.write(content.encode('utf-8'))
    return buffer

##############################################
# Data Loading and Preprocessing Function
##############################################
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data.fillna(data.median(), inplace=True)
    # Handle outliers using percentiles for key columns
    for col in ['avg_bill_value', 'credit_limit', 'investments']:
        lower, upper = data[col].quantile([0.01, 0.99])
        data[col] = data[col].clip(lower, upper)
    # Create credit indicators
    data['credit_utilization'] = data['avg_bill_value'] / (data['credit_limit'] + 1e-5)
    data['financial_stability_score'] = (data['investments'] / (data['avg_bill_value'] + 1)).clip(0, 1)
    return data

##############################################
# Utility: Record explanation history for auditing
##############################################
def record_explanation(instance_id, explanation):
    if 'explanation_history' not in st.session_state:
        st.session_state['explanation_history'] = {}
    st.session_state['explanation_history'][instance_id] = explanation

##############################################
# Sidebar Navigation
##############################################
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Section:", 
                          ("Data Upload & EDA", 
                           "Modeling & Performance", 
                           "Advanced Visualizations", 
                           "Advanced Feature Engineering", 
                           "Dynamic Hyperparameter Tuning", 
                           "Advanced Explainability", 
                           "Interactive Explainability & Model Comparison",
                           "Export Report", 
                           "Advanced What-if Simulation",
                           "Advanced Interpretability Visualizations",
                           "Additional Evaluation Metrics",
                           "Executive Summary with AI",
                           "Virtual Assistant"))

##############################################
# Section 1: Data Upload & EDA
##############################################
if page == "Data Upload & EDA":
    st.title("Data Upload & Exploratory Data Analysis")
    st.markdown("Upload your CSV file containing the data. **Required columns:** client_id, defaulted, avg_bill_value, credit_limit, and investments.")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    
    if uploaded_file is not None:
        if uploaded_file.getbuffer().nbytes == 0:
            st.error("The file is empty. Please upload a valid CSV file.")
        else:
            data = load_data(uploaded_file)
            st.subheader("Credit Indicator Statistics")
            st.write(data[['credit_utilization', 'financial_stability_score']].describe())
            # Optional dynamic filter if 'age' exists in data
            if 'age' in data.columns:
                age_min = int(data['age'].min())
                age_max = int(data['age'].max())
                age_range = st.slider("Filter by Age", age_min, age_max, (age_min, age_max))
                data = data[(data['age'] >= age_range[0]) & (data['age'] <= age_range[1])]
                st.write("Filtered Data:")
                st.write(data.head())
            st.session_state['data'] = data
    else:
        st.info("Please upload the CSV file to continue.")

##############################################
# Section 2: Modeling & Performance
##############################################
if page == "Modeling & Performance":
    st.title("Modeling & Performance")
    if 'data' not in st.session_state:
        st.error("Please upload the data in the 'Data Upload & EDA' section first!")
    else:
        data = st.session_state['data']
        try:
            X = data.drop(columns=['client_id', 'defaulted'])
            y = data['defaulted']
        except KeyError:
            st.error("Make sure the CSV file contains the columns 'client_id' and 'defaulted'.")
            st.stop()
        
        # Data balancing and train/test split using SMOTE
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        X_res = pd.DataFrame(X_res, columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
        
        # Dynamic parameter tuning for RandomForest (example)
        st.sidebar.subheader("RandomForest Parameters")
        n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 200, step=50)
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        
        # Define a set of models
        models = {
            "XGBoost": xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                                         use_label_encoder=False, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "SVC": SVC(probability=True, random_state=42),
            "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
            "GaussianNB": GaussianNB()
        }
        
        # Save models in session_state for later use (for comparison)
        st.session_state['models'] = {}
        results = []
        best_model = None
        best_auc = 0
        
        st.subheader("Model Comparison")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except Exception as e:
                y_proba = model.decision_function(X_test)
            auc = roc_auc_score(y_test, y_proba)
            acc = accuracy_score(y_test, y_pred)
            results.append({"Model": name, "Accuracy": acc, "AUC-ROC": auc})
            st.write(f"Model: {name} | Accuracy: {acc:.3f} | AUC-ROC: {auc:.3f}")
            st.session_state['models'][name] = model
            if auc > best_auc:
                best_auc = auc
                best_model = model
        
        # Save best model for later use
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        result_df = pd.DataFrame(results)
        st.write("Model Comparison Table:")
        st.dataframe(result_df)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc_scores = cross_val_score(best_model, X, y, scoring='roc_auc', cv=cv)
        st.write("Cross-Validated AUC of Best Model:", np.mean(cv_auc_scores))
        
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['best_model'] = best_model
        st.session_state['result_df'] = result_df

##############################################
# Section 3: Advanced Visualizations
##############################################
if page == "Advanced Visualizations":
    st.title("Advanced Visualizations")
    if 'X_test' not in st.session_state:
        st.error("Please complete the modeling in the 'Modeling & Performance' section first!")
    else:
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        best_model = st.session_state['best_model']
        X_train = st.session_state['X_train']
        
        figs_to_report = []  # List for saving figures for the report
        
        # 1. Confusion Matrix using seaborn
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, best_model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["No Default", "Defaulted"],
                    yticklabels=["No Default", "Defaulted"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        figs_to_report.append(fig)
        
        # 2. Interactive ROC Curve with Plotly
        st.subheader("Interactive ROC Curve")
        y_proba_best = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba_best)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Baseline', line=dict(dash='dash')))
        roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(roc_fig)
        
        # 3. Interactive Calibration Curve
        st.subheader("Calibration Curve")
        prob_bins = np.linspace(0, 1, 11)
        bin_ids = np.digitize(y_proba_best, prob_bins) - 1
        bin_true = [y_test[bin_ids == i].mean() if np.sum(bin_ids == i) > 0 else np.nan for i in range(len(prob_bins))]
        calib_fig = px.line(x=prob_bins, y=bin_true, markers=True, 
                            labels={'x': 'Predicted Probability', 'y': 'Observed Frequency'})
        calib_fig.add_scatter(x=[0,1], y=[0,1], mode='lines', name='Ideal', line=dict(dash='dash'))
        calib_fig.update_layout(title="Calibration Curve")
        st.plotly_chart(calib_fig)
        
        # 4. Cumulative Accuracy Profile (CAP)
        st.subheader("Cumulative Accuracy Profile (CAP)")
        df_cap = pd.DataFrame({"y_true": y_test, "y_proba": y_proba_best})
        df_cap.sort_values(by="y_proba", ascending=False, inplace=True)
        df_cap["cum_true"] = df_cap["y_true"].cumsum()
        df_cap["cum_total"] = np.arange(1, len(df_cap)+1)
        total_pos = df_cap["y_true"].sum()
        df_cap["gain"] = df_cap["cum_true"] / total_pos
        cap_fig = go.Figure()
        cap_fig.add_trace(go.Scatter(x=np.linspace(0, 1, len(df_cap)), y=df_cap["gain"], mode='lines', name="CAP"))
        cap_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name="Baseline", line=dict(dash='dash')))
        cap_fig.update_layout(title="Cumulative Accuracy Profile", xaxis_title="Percentile", yaxis_title="Gain")
        st.plotly_chart(cap_fig)
        
        # 5. Distribution of Predicted Scores (Histogram)
        st.subheader("Distribution of Predicted Scores")
        dist_fig = px.histogram(x=y_proba_best, nbins=50, labels={'x': "Probability of Default"}, title="Histogram of Predicted Scores")
        st.plotly_chart(dist_fig)
        
        # 6. Performance Dashboard Table
        st.subheader("Performance Dashboard")
        st.write("Model Metrics Comparison:")
        st.dataframe(st.session_state.get('result_df', pd.DataFrame()))
        
        # 7. SHAP Beeswarm Summary Plot (with sampling and bar plot option)
        st.subheader("SHAP Summary Plot (Beeswarm & Bar)")
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_test)
        
        # Sampling to avoid overcrowding
        max_points = 3000  # adjust as needed
        if len(X_test) > max_points:
            X_test_sample = X_test.sample(n=max_points, random_state=42)
            shap_values_sample = shap_values[[i for i in X_test_sample.index]]
        else:
            X_test_sample = X_test
            shap_values_sample = shap_values

        st.markdown("**Beeswarm Plot (sampled if needed)**")
        shap.summary_plot(shap_values_sample, X_test_sample, max_display=15, show=False)
        fig_beeswarm = plt.gcf()  # Get current figure
        st.pyplot(fig_beeswarm, bbox_inches="tight")
        plt.clf()

        st.markdown("---")
        st.markdown("**Bar Plot (average impact)**")
        shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15, show=False)
        fig_bar = plt.gcf()  # Get current figure
        st.pyplot(fig_bar, bbox_inches="tight")
        plt.clf()

##############################################
# Section 4: Advanced Feature Engineering & Selection
##############################################
if page == "Advanced Feature Engineering":
    st.title("Advanced Feature Engineering & Selection")
    if 'data' not in st.session_state:
        st.error("Please upload the data in the 'Data Upload & EDA' section first!")
    else:
        data = st.session_state['data']
        st.subheader("Select Features for Modeling")
        all_features = list(data.columns)
        # Remove required columns
        required_columns = ['client_id', 'defaulted']
        selectable = [col for col in all_features if col not in required_columns]
        selected_features = st.multiselect("Select variables", selectable, default=selectable)
        
        if len(selected_features) == 0:
            st.error("Please select at least one variable to proceed.")
        else:
            st.write("Selected Features:", selected_features)
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(data[selected_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            
            st.subheader("PCA - Principal Component Analysis")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data[selected_features].dropna())
            pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
            pca_fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA of Selected Data")
            st.plotly_chart(pca_fig)

##############################################
# Section 5: Dynamic Hyperparameter Tuning
##############################################
if page == "Dynamic Hyperparameter Tuning":
    st.title("Dynamic Hyperparameter Tuning")
    if 'data' not in st.session_state:
        st.error("Please upload the data in the 'Data Upload & EDA' section first!")
    else:
        data = st.session_state['data']
        try:
            X = data.drop(columns=['client_id', 'defaulted'])
            y = data['defaulted']
        except KeyError:
            st.error("Check CSV file columns.")
            st.stop()
        
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        X_res = pd.DataFrame(X_res, columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
        
        st.subheader("RandomForest Hyperparameter Tuning")
        n_estimators = st.slider("Number of Trees", 50, 500, 200, step=50)
        max_depth = st.slider("Max Depth", 1, 20, 5)
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:,1]
        except:
            y_proba = model.decision_function(X_test)
        auc = roc_auc_score(y_test, y_proba)
        st.write(f"Model AUC: {auc:.3f}")
        
        # Display Learning Curve
        st.subheader("Learning Curve")
        train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        fig, ax = plt.subplots()
        ax.plot(train_sizes, train_mean, 'o-', label="Training Score")
        ax.plot(train_sizes, test_mean, 'o-', label="Validation Score")
        ax.set_xlabel("Training Size")
        ax.set_ylabel("Score")
        ax.set_title("Learning Curve")
        ax.legend()
        st.pyplot(fig)

##############################################
# Section 6: Advanced Explainability
##############################################
if page == "Advanced Explainability":
    st.title("Advanced Model Explainability")
    if 'X_test' not in st.session_state:
        st.error("Please complete the modeling in the 'Modeling & Performance' section first!")
    else:
        X_test = st.session_state['X_test']
        best_model = st.session_state['best_model']
        X_train = st.session_state['X_train']
        
        st.subheader("SHAP Dependence Plot")
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_test)
        feature = st.selectbox("Select a feature for the Dependence Plot", X_test.columns)
        shap.dependence_plot(feature, shap_values.values, X_test, show=False)
        fig_dep = plt.gcf()
        st.pyplot(fig_dep, bbox_inches="tight")
        plt.clf()
        
        st.subheader("SHAP Force Plot for a Single Instance")
        instance_index = st.slider("Select instance for Force Plot", 0, X_test.shape[0]-1, 0)
        force_fig = shap.force_plot(explainer.expected_value, shap_values.values[instance_index], 
                                      X_test.iloc[instance_index], matplotlib=True, show=False)
        st.pyplot(force_fig, bbox_inches="tight")
        plt.clf()
        
        st.subheader("Local Explanation with LIME and History Logging")
        if 'lime' in globals():
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values, 
                feature_names=X_train.columns, 
                class_names=["No Default", "Defaulted"], 
                discretize_continuous=True
            )
            i = st.slider("Select instance for LIME explanation", 0, X_test.shape[0]-1, 0)
            exp = lime_explainer.explain_instance(
                X_test.iloc[i].values, 
                best_model.predict_proba, 
                num_features=10
            )
            st.write("LIME Explanation:", exp.as_list())
            record_explanation(i, exp.as_list())
            st.subheader("Explanation History")
            st.write(st.session_state.get('explanation_history', {}))
        else:
            st.info("LIME is not installed. To use LIME, install with: pip install lime")

##############################################
# Section 7: Interactive Explainability & Model Comparison
##############################################
if page == "Interactive Explainability & Model Comparison":
    st.title("Interactive Explainability & Model Comparison")
    if 'models' not in st.session_state or len(st.session_state['models']) < 2:
        st.error("At least two models must be trained in the 'Modeling & Performance' section for comparison!")
    else:
        models = st.session_state['models']
        selected_models = st.multiselect("Select models for comparison", list(models.keys()))
        if selected_models:
            cols = st.columns(len(selected_models))
            for idx, model_name in enumerate(selected_models):
                model = models[model_name]
                # Use TreeExplainer for tree-based models
                if model.__class__.__name__ in ["RandomForestClassifier", "GradientBoostingClassifier", "AdaBoostClassifier", "XGBClassifier"]:
                    explainer = shap.TreeExplainer(model)
                elif model.__class__.__name__ == "SVC":
                    # For SVC, use KernelExplainer with a sample of the training data as background
                    background = st.session_state['X_train'].iloc[:100]
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                else:
                    explainer = shap.Explainer(model, st.session_state['X_train'])
                
                shap_values = explainer(st.session_state['X_test'])
                with cols[idx]:
                    st.subheader(f"SHAP Summary - {model_name}")
                    shap.summary_plot(shap_values, st.session_state['X_test'], max_display=10, show=False)
                    st.pyplot(plt.gcf(), bbox_inches="tight")
                    plt.clf()

##############################################
# Section 8: Export Report
##############################################
if page == "Export Report":
    st.title("Export Report")
    if 'result_df' not in st.session_state or 'best_model' not in st.session_state:
        st.error("Please complete the modeling and analyses before generating the report!")
    else:
        metrics = {
            "AUC": roc_auc_score(st.session_state['y_test'], st.session_state['best_model'].predict_proba(st.session_state['X_test'])[:,1]),
            "KS": np.max(np.abs(np.linspace(0,1,len(st.session_state['X_test'])) - 
                                  np.sort(st.session_state['best_model'].predict_proba(st.session_state['X_test'])[:,1])))
        }
        report_text = "<p>This report was generated from the interactive Credit Risk Dashboard.</p>"
        figs_list = []  # Optionally, add figures from previous sections here
        st.markdown("Click the button below to download the HTML report.")
        report_buffer = generate_report(st.session_state.get('result_df', pd.DataFrame()), metrics, report_text, figs_list)
        st.download_button("Download Report", data=report_buffer, file_name="credit_risk_report.html", mime="text/html")

##############################################
# Section 9: Advanced What-if Simulation
##############################################
if page == "Advanced What-if Simulation":
    st.title("Advanced What-if Simulation")
    if 'X_test' not in st.session_state:
        st.error("Please complete the modeling in the 'Modeling & Performance' section first!")
    else:
        X_test = st.session_state['X_test']
        best_model = st.session_state['best_model']
        
        st.markdown("### Multiple Scenario Simulation")
        num_scenarios = st.number_input("Number of scenarios to simulate", min_value=1, max_value=5, value=1)
        scenarios = []
        
        for i in range(num_scenarios):
            st.markdown(f"#### Scenario {i+1}")
            credit_util_input = st.number_input(f"Scenario {i+1} - Credit Utilization", 
                                                value=float(X_test["credit_utilization"].mean()), key=f"cu_{i}")
            financial_stability_input = st.number_input(f"Scenario {i+1} - Financial Stability Score", 
                                                         value=float(X_test["financial_stability_score"].mean()), key=f"fs_{i}")
            # Create a copy of a reference instance and update selected features for simulation
            scenario_instance = X_test.iloc[0].copy()
            scenario_instance["credit_utilization"] = credit_util_input
            scenario_instance["financial_stability_score"] = financial_stability_input
            scenarios.append(scenario_instance)
        
        st.markdown("### Predicted Default Probabilities for Each Scenario")
        for idx, scenario in enumerate(scenarios):
            proba = best_model.predict_proba(scenario.values.reshape(1, -1))[:, 1][0]
            st.write(f"Scenario {idx+1}: Predicted Default Probability = {proba:.2f}")
        
        st.markdown("### Sensitivity Analysis - Visual Feedback")
        feature = st.selectbox("Select feature for sensitivity analysis", X_test.columns)
        instance_ref = X_test.iloc[0].copy()
        base_val = instance_ref[feature]
        values = np.linspace(base_val * 0.5, base_val * 1.5, 20)
        probas = []
        for val in values:
            instance_ref[feature] = val
            proba = best_model.predict_proba(instance_ref.values.reshape(1, -1))[:, 1][0]
            probas.append(proba)
        sens_fig = go.Figure()
        sens_fig.add_trace(go.Scatter(x=values, y=probas, mode='lines+markers', name='Sensitivity'))
        sens_fig.update_layout(title=f"Sensitivity of Prediction to {feature}",
                               xaxis_title=feature, yaxis_title="Predicted Default Probability")
        st.plotly_chart(sens_fig)

##############################################
# Section 10: Advanced Interpretability Visualizations
##############################################
if page == "Advanced Interpretability Visualizations":
    st.title("Advanced Interpretability Visualizations")
    if 'X_test' not in st.session_state or 'best_model' not in st.session_state:
        st.error("Please complete the modeling in the 'Modeling & Performance' section first!")
    else:
        X_test = st.session_state['X_test']
        best_model = st.session_state['best_model']
        X_train = st.session_state['X_train']
        
        st.header("Partial Dependence Plot (PDP) and ICE")
        # Allow user to select one or more features for PDP/ICE
        features = st.multiselect("Select features for PDP/ICE", X_test.columns, default=[X_test.columns[0]])
        if features:
            try:
                disp = PartialDependenceDisplay.from_estimator(best_model, X_test, features, kind="both", subsample=50, grid_resolution=20)
                st.pyplot(disp.figure_, bbox_inches="tight")
            except Exception as e:
                st.error(f"Error generating PDP/ICE: {e}")

        st.header("Waterfall Chart for a Single Prediction")
        instance_idx = st.slider("Select instance for Waterfall Chart", 0, X_test.shape[0]-1, 0)
        st.markdown("Waterfall Chart for the selected instance:")
        try:
            explainer = shap.Explainer(best_model, X_train)
            shap_values = explainer(X_test)
            shap.waterfall_plot(shap_values[instance_idx])
            fig_waterfall = plt.gcf()  # Get the current figure
            st.pyplot(fig_waterfall, bbox_inches="tight")
            plt.clf()
        except Exception as e:
            st.error(f"Error generating Waterfall Chart: {e}")

        st.header("Permutation Feature Importance")
        from sklearn.inspection import permutation_importance
        result = permutation_importance(best_model, X_test, st.session_state['y_test'], n_repeats=10, random_state=42, scoring='roc_auc')
        sorted_idx = result.importances_mean.argsort()
        fig_perm, ax_perm = plt.subplots()
        ax_perm.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
        ax_perm.set_title("Permutation Feature Importance")
        st.pyplot(fig_perm, bbox_inches="tight")
        plt.clf()

        st.header("Interactive Dashboard with Dynamic Filters")
        st.markdown("Use the filters below to visualize a subset of the test data and model predictions.")
        # Example filters using credit_utilization and financial_stability_score
        cu_min, cu_max = float(X_test["credit_utilization"].min()), float(X_test["credit_utilization"].max())
        fs_min, fs_max = float(X_test["financial_stability_score"].min()), float(X_test["financial_stability_score"].max())
        cu_filter = st.slider("Credit Utilization", cu_min, cu_max, (cu_min, cu_max))
        fs_filter = st.slider("Financial Stability Score", fs_min, fs_max, (fs_min, fs_max))
        filtered_data = X_test[(X_test["credit_utilization"] >= cu_filter[0]) & (X_test["credit_utilization"] <= cu_filter[1]) &
                                (X_test["financial_stability_score"] >= fs_filter[0]) & (X_test["financial_stability_score"] <= fs_filter[1])]
        st.write("Filtered Test Data (first 5 rows):", filtered_data.head())
        if not filtered_data.empty:
            proba_filtered = best_model.predict_proba(filtered_data)[:,1]
            fig_filter = px.histogram(x=proba_filtered, nbins=30, labels={'x': "Predicted Default Probability"}, title="Prediction Distribution for Filtered Data")
            st.plotly_chart(fig_filter)

        st.header("Multivariate Sensitivity Analysis (Heatmap)")
        st.markdown("Select two features to analyze their combined effect on the model prediction.")
        feat1 = st.selectbox("Select first feature", X_test.columns, index=0)
        feat2 = st.selectbox("Select second feature", X_test.columns, index=1)
        # Define range for each feature using quantiles
        feat1_vals = np.linspace(X_test[feat1].quantile(0.05), X_test[feat1].quantile(0.95), 20)
        feat2_vals = np.linspace(X_test[feat2].quantile(0.05), X_test[feat2].quantile(0.95), 20)
        heatmap_data = np.zeros((len(feat2_vals), len(feat1_vals)))
        instance_base = X_test.iloc[0].copy()
        for i, val1 in enumerate(feat1_vals):
            for j, val2 in enumerate(feat2_vals):
                instance_temp = instance_base.copy()
                instance_temp[feat1] = val1
                instance_temp[feat2] = val2
                pred = best_model.predict_proba(instance_temp.values.reshape(1, -1))[:, 1][0]
                heatmap_data[j, i] = pred
        fig_heat = px.imshow(heatmap_data, x=feat1_vals, y=feat2_vals, 
                             labels={'x': feat1, 'y': feat2, 'color': "Predicted Default Probability"},
                             title="Multivariate Sensitivity Analysis")
        st.plotly_chart(fig_heat)

##############################################
# Section 11: Additional Evaluation Metrics
##############################################
if page == "Additional Evaluation Metrics":
    st.title("Additional Evaluation Metrics")
    if 'X_test' not in st.session_state or 'y_test' not in st.session_state or 'best_model' not in st.session_state:
        st.error("Please complete the modeling in the 'Modeling & Performance' section first!")
    else:
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        best_model = st.session_state['best_model']
        
        st.header("Lift Curve")
        # Calculate lift curve data
        y_proba = best_model.predict_proba(X_test)[:,1]
        sorted_idx = np.argsort(-y_proba)
        sorted_y = y_test.iloc[sorted_idx].reset_index(drop=True)
        cum_positives = np.cumsum(sorted_y)
        total_positives = sorted_y.sum()
        percentile = np.linspace(1/len(sorted_y), 1.0, len(sorted_y))
        gain = cum_positives / total_positives
        # Avoid division by zero for the first percentile
        lift = gain / percentile
        fig_lift, ax_lift = plt.subplots()
        ax_lift.plot(percentile, lift, label='Lift Curve')
        ax_lift.set_xlabel("Percentile")
        ax_lift.set_ylabel("Lift")
        ax_lift.set_title("Lift Curve")
        st.pyplot(fig_lift, bbox_inches="tight")
        plt.clf()

        st.header("Learning Curve with Bias/Variance Analysis")
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, test_scores = learning_curve(best_model, X_test, y_test, cv=5, train_sizes=np.linspace(0.1, 1.0, 5))
        train_std = np.std(train_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        fig_lv, ax_lv = plt.subplots()
        ax_lv.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training Score")
        ax_lv.fill_between(train_sizes, np.mean(train_scores, axis=1)-train_std, np.mean(train_scores, axis=1)+train_std, alpha=0.2, color="r")
        ax_lv.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Validation Score")
        ax_lv.fill_between(train_sizes, np.mean(test_scores, axis=1)-test_std, np.mean(test_scores, axis=1)+test_std, alpha=0.2, color="g")
        ax_lv.set_xlabel("Training Size")
        ax_lv.set_ylabel("Score")
        ax_lv.set_title("Learning Curve with Bias-Variance Analysis")
        ax_lv.legend()
        st.pyplot(fig_lv, bbox_inches="tight")
        plt.clf()

        st.header("Bootstrap Confidence Intervals for Mean Predictions")
        # Use bootstrapping to estimate the confidence interval of mean predictions
        n_bootstraps = 100
        boot_means = []
        rng = np.random.RandomState(42)
        for i in range(n_bootstraps):
            indices = rng.choice(range(len(X_test)), size=len(X_test), replace=True)
            boot_X = X_test.iloc[indices]
            boot_pred = best_model.predict_proba(boot_X)[:,1]
            boot_means.append(np.mean(boot_pred))
        boot_means = np.array(boot_means)
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)
        st.write("Bootstrap Confidence Interval for Mean Predictions (95%): [{:.3f}, {:.3f}]".format(ci_lower, ci_upper))
        fig_ci = px.histogram(boot_means, nbins=20, title="Bootstrap Distribution of Mean Predictions")
        st.plotly_chart(fig_ci)

##############################################
# Section 12: Executive Summary with AI
##############################################
if page == "Executive Summary with AI":
    st.title("Executive Summary with AI")
    if 'result_df' not in st.session_state or 'best_model' not in st.session_state:
        st.error("Please complete the modeling and analyses before generating the executive summary!")
    else:
        # Aggregate summarized information from the results
        metrics_info = "Model Performance (Metrics):\n" + str(st.session_state['result_df'].to_dict()) + "\n"
        shap_insights = (
            "Feature Insights (SHAP):\n"
            "Analyses indicate that 'credit_utilization' and 'financial_stability_score' are critical for predicting default risk. "
            "The SHAP plots show that lower utilization and higher stability are associated with lower default risk.\n"
        )
        learning_curve_info = (
            "Learning Curve: The models show stable learning curves with good cross-validation and high AUC, indicating strong performance.\n"
        )
        sensitivity_info = (
            "Sensitivity Analysis: Changes in key indicators significantly affect the default predictions.\n"
        )
        
        # Summarize client data by counts (instead of listing all IDs)
        client_info = ""
        if 'data' in st.session_state:
            data = st.session_state['data']
            if 'client_id' in data.columns:
                total_clients = len(data)
                approved_count = len(data.loc[(data['credit_utilization'] < 0.35) & (data['financial_stability_score'] > 0.80)])
                rejected_count = total_clients - approved_count
                client_info = (
                    f"Client Summary:\nTotal Clients: {total_clients}. "
                    f"Approved Clients (credit_utilization < 0.35 and financial_stability_score > 0.80): {approved_count}. "
                    f"Rejected Clients: {rejected_count}.\n"
                )
        
        # Let the user choose the summary type
        summary_type_option = st.radio("Select Summary Type", ("Technical Report", "Operational Report"))
        
        # Function to generate a detailed prompt with clear sections and chain-of-thought instructions
        def generate_detailed_prompt(summary_type, metrics_info, shap_insights, learning_curve_info, sensitivity_info, client_info):
            if summary_type == "Technical Report":
                prompt = (
                    "You are a senior credit risk analyst. Please follow these steps:\n"
                    "1. List the main points regarding model performance based on the provided metrics.\n"
                    "2. Describe in detail the insights from the SHAP analysis, learning curves, and sensitivity analysis.\n"
                    "3. Based on these data, generate an in-depth technical report that explains why clients with low credit utilization "
                    "and high financial stability should be approved for credit.\n\n"
                    "=== Model Performance (Metrics) ===\n" + metrics_info + "\n"
                    "=== Feature Insights (SHAP) ===\n" + shap_insights + "\n"
                    "=== Learning Curve and Sensitivity Analysis ===\n" + learning_curve_info + sensitivity_info + "\n"
                    "=== Credit Recommendations ===\n"
                    "Explain in technical detail the reasons for granting credit to clients with low credit utilization and high financial stability.\n"
                )
            else:  # Operational Report
                prompt = (
                    "You are a senior credit risk analyst. Please follow these steps:\n"
                    "1. Summarize the client data, highlighting the number of approved and rejected clients based on the criteria.\n"
                    "2. List the key criteria used for approval (e.g., credit_utilization < 0.35 and financial_stability_score > 0.80).\n"
                    "3. Generate a concise operational report that provides practical guidelines for credit approval, summarizing the information without listing individual client IDs.\n\n"
                    "=== Client Summary ===\n" + client_info + "\n"
                    "=== Approval Criteria and Recommendations ===\n"
                    "Provide practical operational guidelines for credit approval based on the above criteria.\n"
                )
            return prompt
        
        final_prompt = generate_detailed_prompt(
            summary_type_option,
            metrics_info,
            shap_insights,
            learning_curve_info,
            sensitivity_info,
            client_info
        )
        
        st.markdown("### Generated Prompt:")
        st.code(final_prompt)
        
        with st.spinner("Generating executive summary using AI..."):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a senior credit risk analyst."},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.5
                )
                executive_summary = response.choices[0].message.content.strip()
                st.subheader("Executive Summary Generated by AI")
                st.write(executive_summary)
            except Exception as e:
                st.error(f"Error generating summary: {e}")

##############################################
# Section 13: Virtual Assistant with GPT
##############################################
if page == "Virtual Assistant":
    st.title("Virtual Assistant - Project Questions")
    st.markdown(
        "Ask questions about any aspect of the project. For example: **'Should I approve client 123? Why or why not?'** or **'Which model is best for X?'**"
    )
    
    user_question = st.text_input("Type your question here:")
    
    if st.button("Submit Question"):
        if not user_question.strip():
            st.error("Please enter a valid question.")
        else:
            # Build a comprehensive summary of project data from session_state
            project_context = ""
            
            # Basic Data Information
            if 'data' in st.session_state:
                data = st.session_state['data']
                total_clients = len(data)
                project_context += f"Total clients in dataset: {total_clients}.\n"
                if 'client_id' in data.columns:
                    project_context += "Client IDs are available.\n"
                # Example approval criteria (if defined in your project)
                project_context += "Approval Criteria: credit_utilization < 0.35 and financial_stability_score > 0.80.\n"
                # Include basic statistics for key indicators
                if 'credit_utilization' in data.columns and 'financial_stability_score' in data.columns:
                    avg_cu = data['credit_utilization'].mean()
                    avg_fs = data['financial_stability_score'].mean()
                    project_context += f"Average Credit Utilization: {avg_cu:.3f}.\n"
                    project_context += f"Average Financial Stability Score: {avg_fs:.3f}.\n"
                    
                # Check if the question mentions a specific client (e.g., "client 123")
                import re
                client_match = re.search(r'client\s*(\d+)', user_question, re.IGNORECASE)
                if client_match:
                    client_id = int(client_match.group(1))
                    client_data = data[data['client_id'] == client_id]
                    if not client_data.empty:
                        client_details = client_data.to_dict(orient='records')[0]
                        project_context += f"Data for Client {client_id}: " + str(client_details) + "\n"
                    else:
                        project_context += f"No data found for Client {client_id}.\n"
            
            # Model Performance Data
            if 'result_df' in st.session_state:
                result_df = st.session_state['result_df']
                project_context += "Model Performance Metrics:\n" + str(result_df.to_dict()) + "\n"
            
            # Best Model and Prediction Information
            if 'X_test' in st.session_state and 'best_model' in st.session_state:
                best_model = st.session_state['best_model']
                X_test = st.session_state['X_test']
                try:
                    proba = best_model.predict_proba(X_test)[:, 1]
                    avg_proba = np.mean(proba)
                    project_context += f"Average Predicted Default Probability: {avg_proba:.3f}.\n"
                except Exception as e:
                    project_context += "Could not compute average predicted probability.\n"
            
            # Additional Data: Explanation History if available
            if 'explanation_history' in st.session_state:
                explanations = st.session_state['explanation_history']
                if explanations:
                    project_context += f"Number of recorded local explanations: {len(explanations)}.\n"
            
            # Additional Models info if available
            if 'models' in st.session_state:
                models = st.session_state['models']
                model_names = list(models.keys())
                project_context += f"Trained models: {', '.join(model_names)}.\n"
            
            # Compile the final prompt with the project context
            prompt = (
                "You are a virtual assistant specialized in credit risk analysis with full access to the project data. "
                "Below is a detailed summary of the project context, including client data, approval criteria, model performance, "
                "and other relevant metrics:\n\n"
                f"{project_context}\n"
                "Using this information, provide a detailed, data-driven answer to the following question:\n"
                f"'{user_question}'."
            )
            
            with st.spinner("Generating answer..."):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a virtual assistant specialized in credit risk analysis with full access to the project data."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.5
                    )
                    answer = response.choices[0].message.content.strip()
                    st.subheader("Virtual Assistant Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
