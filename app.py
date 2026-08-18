import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classification Dashboard", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_FILES = {
    "Logistic Regression": "model/Logistic_Regression.pkl",
    "Decision Tree":        "model/Decision_Tree.pkl",
    "kNN":                  "model/kNN.pkl",
    "Naive Bayes":          "model/Naive_Bayes.pkl",
    "Random Forest":        "model/Random_Forest.pkl",
}
SCALED_MODELS = {"Logistic Regression", "kNN", "Naive Bayes"}
TARGET_COL = "income"

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.pkl")

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "F1":        round(f1_score(y_true, y_pred), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
    }

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["<=50K", ">50K"],
                yticklabels=["<=50K", ">50K"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("ML Classification Dashboard")
st.sidebar.markdown("**Dataset:** Adult Income (UCI)")
st.sidebar.markdown("**Task:** Binary Classification (Income >50K?)")
st.sidebar.divider()

DATA_FILE = "test_data.csv"

selected_model = st.sidebar.selectbox(
    "Select Model", list(MODEL_FILES.keys())
)

compare_all = st.sidebar.checkbox("Compare All Models", value=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("ML Classification Dashboard")
st.markdown("### Adult Income Dataset — 5 Classifier Comparison")

# Automatically load the dataset bundled with the Streamlit project.
# Keep test_data.csv in the same repository folder as app.py.
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    st.error(
        "Dataset file `test_data.csv` was not found. "
        "Please place `test_data.csv` in the same folder as `app.py`."
    )
    st.stop()
if TARGET_COL not in df.columns:
    st.error(f"Target column `{TARGET_COL}` not found in uploaded CSV.")
    st.stop()

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

st.markdown(
    f"**Rows loaded:** {len(df)} &nbsp;&nbsp; "
    f"**Features:** {X.shape[1]} &nbsp;&nbsp; "
    f"**Target:** `{TARGET_COL}`"
)

# ── Dataset overview ──────────────────────────────────────────────────────────
with st.expander("Dataset Overview", expanded=True):
    tab1, tab2, tab3 = st.tabs(["Dataset Preview", "Summary", "Class Distribution"])

    with tab1:
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", len(df))
        c2.metric("Features", X.shape[1])
        c3.metric("Missing Values", int(df.isna().sum().sum()))
        c4.metric("Target Classes", df[TARGET_COL].nunique())

        summary = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(df[c].dtype) for c in df.columns],
            "Missing": [int(df[c].isna().sum()) for c in df.columns],
            "Unique Values": [int(df[c].nunique()) for c in df.columns]
        })
        st.dataframe(summary, use_container_width=True)

    with tab3:
        class_counts = df[TARGET_COL].value_counts().sort_index()
        class_df = pd.DataFrame({
            "Income Class": ["<=50K", ">50K"][:len(class_counts)],
            "Count": class_counts.values
        })
        st.bar_chart(class_df.set_index("Income Class"))

scaler = load_scaler()

# ── Single model view ─────────────────────────────────────────────────────────
if not compare_all:
    st.subheader(f"Results — {selected_model}")

    model = load_model(MODEL_FILES[selected_model])
    X_input = scaler.transform(X) if selected_model in SCALED_MODELS else X.values

    y_pred = model.predict(X_input)
    y_prob = model.predict_proba(X_input)[:, 1]
    metrics = compute_metrics(y, y_pred, y_prob)

    # Metric cards
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, v)

    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(y, y_pred, selected_model)
        st.pyplot(fig)

    with col2:
        st.subheader("Classification Report")
        report = classification_report(
            y, y_pred, target_names=["<=50K", ">50K"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4))

# ── Compare all models ────────────────────────────────────────────────────────
else:
    st.subheader("All Models — Side-by-Side Comparison")

    all_metrics = []
    all_preds = {}

    for name, path in MODEL_FILES.items():
        model = load_model(path)
        X_input = scaler.transform(X) if name in SCALED_MODELS else X.values
        y_pred = model.predict(X_input)
        y_prob = model.predict_proba(X_input)[:, 1]
        m = compute_metrics(y, y_pred, y_prob)
        m["Model"] = name
        all_metrics.append(m)
        all_preds[name] = (y_pred, y_prob)

    results_df = pd.DataFrame(all_metrics).set_index("Model")
    st.dataframe(results_df.style.highlight_max(axis=0, color="#d4edda")
                                  .highlight_min(axis=0, color="#f8d7da"), use_container_width=True)

    st.divider()
    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for i, (name, (y_pred, _)) in enumerate(all_preds.items()):
        with cols[i % 3]:
            fig = plot_confusion_matrix(y, y_pred, name)
            st.pyplot(fig)

    st.divider()
    st.subheader("Metric Comparison Chart")
    metric_choice = st.selectbox("Metric to visualize", ["Accuracy", "AUC", "F1", "MCC", "Precision", "Recall"])
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bars = ax2.bar(results_df.index, results_df[metric_choice], color=sns.color_palette("Blues_d", len(results_df)))
    ax2.set_ylabel(metric_choice)
    ax2.set_title(f"{metric_choice} by Model")
    ax2.set_ylim(0, 1)
    for bar, val in zip(bars, results_df[metric_choice]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig2)

st.sidebar.divider()
st.sidebar.markdown("M.Tech AIML/DSE — ML Assignment 2")
