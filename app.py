import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                              roc_auc_score, precision_score, recall_score,
                              f1_score, matthews_corrcoef)

st.set_page_config(page_title="Classification Model Explorer", layout="wide")
st.title("Classification Model Explorer")   # personalize this

TARGET = 'your_target_column'   # must match train_models.py

@st.cache_resource
def load_artifacts():
    scaler = joblib.load('model/scaler.pkl')
    feature_cols = joblib.load('model/feature_columns.pkl')
    models = {
        'Logistic Regression': joblib.load('model/logistic_regression.pkl'),
        'Decision Tree'      : joblib.load('model/decision_tree.pkl'),
        'kNN'                : joblib.load('model/knn.pkl'),
        'Naive Bayes'        : joblib.load('model/naive_bayes.pkl'),
        'Random Forest'      : joblib.load('model/random_forest.pkl'),
    }
    return scaler, feature_cols, models

scaler, feature_cols, models = load_artifacts()

# a. Dataset upload
uploaded = st.file_uploader("Upload test CSV", type=['csv'])

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.write("Preview:", data.head())

    y_true = data[TARGET]
    X_new = pd.get_dummies(data.drop(columns=[TARGET]), drop_first=True)
    X_new = X_new.reindex(columns=feature_cols, fill_value=0)  # aligns columns safely
    X_new_sc = scaler.transform(X_new)

    # b. Model selection dropdown
    model_name = st.selectbox("Choose a model", list(models.keys()))
    model = models[model_name]

    y_pred  = model.predict(X_new_sc)
    y_proba = model.predict_proba(X_new_sc)
    is_binary = len(np.unique(y_true)) == 2
    avg = 'binary' if is_binary else 'weighted'

    # c. Evaluation metrics
    st.subheader(f"Metrics — {model_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
    c1.metric("Precision", f"{precision_score(y_true, y_pred, average=avg, zero_division=0):.3f}")
    c2.metric("Recall", f"{recall_score(y_true, y_pred, average=avg, zero_division=0):.3f}")
    c2.metric("F1 Score", f"{f1_score(y_true, y_pred, average=avg, zero_division=0):.3f}")
    auc = (roc_auc_score(y_true, y_proba[:, 1]) if is_binary
           else roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted'))
    c3.metric("AUC", f"{auc:.3f}")
    c3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.3f}")

    # d. Confusion matrixs
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred, zero_division=0))
else:
    st.info("Upload a test CSV to see model results.")
