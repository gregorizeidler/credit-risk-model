import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, roc_curve, 
                             precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

st.title("Credit Risk Analysis Dashboard")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

def load_data(file):
    df = pd.read_csv(file)
    df.fillna(df.median(), inplace=True)
    df['credit_utilization'] = df['avg_bill_value'] / (df['credit_limit'] + 1e-5)
    return df

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Sample Data:", data.head())

    # Splitting Data
    X = data.drop(columns=['client_id', 'defaulted'])
    y = data['defaulted']
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train multiple models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVC": SVC(probability=True, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        "GaussianNB": GaussianNB()
    }

    results = []
    best_model = None
    best_auc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc, "AUC-ROC": auc})
        if auc > best_auc:
            best_auc = auc
            best_model = model
    
    # Save best model
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Display Metrics
    results_df = pd.DataFrame(results)
    st.write("Model Performance Comparison:", results_df)

    # Confusion Matrix
    st.subheader("Confusion Matrix - Best Model")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax)
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve - Best Model")
    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline', line=dict(dash='dash')))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc)

    # SHAP Explainability
    st.subheader("Feature Importance using SHAP")
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(bbox_inches="tight")

    # Save report
    report_text = "Best model: " + str(best_model) + " with AUC-ROC: " + str(best_auc)
    with open("reports/model_performance.txt", "w") as f:
        f.write(report_text)
