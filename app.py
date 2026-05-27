import os
import pickle

import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Mall Customer KNN Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# ---------------- LOAD CSS ---------------- #

def load_css(file):
    with open(file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# ---------------- TITLE ---------------- #

st.markdown("""
<div class="main-title">
    <h1>🛍️ Mall Customer Classification Dashboard</h1>
    <p>Predict Customer Category using <b>K-Nearest Neighbors (KNN)</b></p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATASET ---------------- #

@st.cache_data
def load_data():

    df = pd.read_csv("data/Mall_Customers.csv")

    return df

df = load_data()

# ---------------- FIX COLUMN ISSUE ---------------- #
# Dataset column is "Genre", not "Gender"

encoder = LabelEncoder()

df["Genre_Encoded"] = encoder.fit_transform(
    df["Genre"]
)

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("⚙️ Dashboard Settings")

show_data = st.sidebar.checkbox(
    "Show Dataset",
    True
)

show_stats = st.sidebar.checkbox(
    "Show Statistics",
    True
)

show_heatmap = st.sidebar.checkbox(
    "Show Correlation Heatmap",
    True
)

k_value = st.sidebar.slider(
    "Select K Value",
    min_value=1,
    max_value=20,
    value=5
)

weights = st.sidebar.selectbox(
    "Weights",
    ["uniform", "distance"],
    index=0
)

algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["auto", "ball_tree", "kd_tree", "brute"],
    index=0
)

leaf_size = st.sidebar.slider(
    "Leaf Size",
    min_value=10,
    max_value=50,
    value=30
)

metric = st.sidebar.selectbox(
    "Distance Metric",
    ["minkowski", "euclidean", "manhattan", "chebyshev"],
    index=0
)

p = st.sidebar.slider(
    "Power Parameter (p)",
    min_value=1,
    max_value=5,
    value=2
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05
)

random_state = st.sidebar.slider(
    "Random State",
    1,
    100,
    42
)

# ---------------- DATASET DISPLAY ---------------- #

st.markdown("---")

if show_data:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

if show_stats:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📊 Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# ---------------- FEATURES & TARGET ---------------- #

x = df[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

y = df["Genre_Encoded"]

# ---------------- TRAIN TEST SPLIT ---------------- #

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

# ---------------- SCALING ---------------- #

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(
    x_train
)

x_test_scaled = scaler.transform(
    x_test
)

# ---------------- MODEL ---------------- #

model = KNeighborsClassifier(
    n_neighbors=k_value,
    weights=weights,
    algorithm=algorithm,
    leaf_size=leaf_size,
    metric=metric,
    p=p
)

model.fit(
    x_train_scaled,
    y_train
)

# ---------------- SAVE MODEL ---------------- #

model_dir = "models"

os.makedirs(
    model_dir,
    exist_ok=True
)

with open(
    os.path.join(model_dir, "knn_model.pkl"),
    "wb"
) as f:
    pickle.dump(model, f)

with open(
    os.path.join(model_dir, "scaler.pkl"),
    "wb"
) as f:
    pickle.dump(scaler, f)

with open(
    os.path.join(model_dir, "label_encoder.pkl"),
    "wb"
) as f:
    pickle.dump(encoder, f)

# ---------------- PREDICTIONS ---------------- #

y_pred = model.predict(
    x_test_scaled
)

# ---------------- METRICS ---------------- #

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

# ---------------- PERFORMANCE ---------------- #

st.markdown("## 🚀 Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Accuracy",
    f"{accuracy:.2f}"
)

c2.metric(
    "Precision",
    f"{precision:.2f}"
)

c3.metric(
    "Recall",
    f"{recall:.2f}"
)

c4.metric(
    "F1 Score",
    f"{f1:.2f}"
)

# ---------------- VISUALIZATION ---------------- #

st.markdown("---")

col1, col2 = st.columns(2)

# -------- SCATTER PLOT -------- #

with col1:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📉 Income vs Spending Score")

    fig = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Genre",
        size="Age",
        hover_data=["CustomerID"],
        title="Customer Distribution"
    )

    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# -------- CONFUSION MATRIX -------- #

with col2:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig2, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig2)

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# ---------------- HEATMAP ---------------- #

if show_heatmap:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    corr = numeric_df.corr()

    fig3, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig3)

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# ---------------- CLASSIFICATION REPORT ---------------- #

st.markdown("---")

st.markdown("""
<div class="card">
<h3>🧠 Classification Report</h3>
</div>
""", unsafe_allow_html=True)

report = classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
)

st.text(report)

# ---------------- PREDICTION SECTION ---------------- #

st.markdown("---")

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("🎯 Predict Customer Category")

age = st.slider(
    "Age",
    int(df["Age"].min()),
    int(df["Age"].max()),
    30
)

income = st.slider(
    "Annual Income (k$)",
    int(df["Annual Income (k$)"].min()),
    int(df["Annual Income (k$)"].max()),
    60
)

score = st.slider(
    "Spending Score (1-100)",
    int(df["Spending Score (1-100)"].min()),
    int(df["Spending Score (1-100)"].max()),
    50
)

# ---------------- INPUT DATA ---------------- #

input_data = np.array([
    [age, income, score]
])

# ---------------- SCALE INPUT ---------------- #

input_scaled = scaler.transform(
    input_data
)

# ---------------- PREDICTION ---------------- #

prediction = model.predict(
    input_scaled
)[0]

probability = np.max(
    model.predict_proba(input_scaled)
)

predicted_genre = encoder.inverse_transform(
    [prediction]
)[0]

# ---------------- RESULT ---------------- #

st.markdown(f"""
<div class="prediction-box">
    🛍️ Predicted Customer Category
    <br><br>
    <b>{predicted_genre}</b>
    <br><br>
    📌 Confidence:
    <b>{probability:.2%}</b>
</div>
""", unsafe_allow_html=True)

st.progress(float(probability))

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ---------------- DOWNLOAD PREDICTIONS ---------------- #

st.markdown("---")

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("📥 Download Predictions")

pred_df = pd.DataFrame({
    "Actual": encoder.inverse_transform(y_test),
    "Predicted": encoder.inverse_transform(y_pred)
})

csv = pred_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="mall_customer_predictions.csv",
    mime="text/csv"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

