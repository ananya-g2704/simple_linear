import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- Page Config ----------------
st.set_page_config("Logistic Regression", layout="centered")

# ---------------- Load CSS ----------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style_log.css")

# ---------------- Title ----------------
st.markdown("""
<div class="card">
    <h1>Logistic Regression</h1>
    <p>Predict <b>Smoker Status</b> using customer details (Binary Classification)</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# ---------------- Dataset Preview ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# ✅ DATA PREPARATION (CORRECT ORDER)
# ==================================================

# Target variable (Binary)
y = df["smoker"].map({"Yes": 1, "No": 0})

# Features
X = df.drop("smoker", axis=1)

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ---------------- Train Model ----------------
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# ---------------- Metrics ----------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ---------------- Confusion Matrix ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Model Performance ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{accuracy:.2f}")
c2.metric("Error Rate", f"{1 - accuracy:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Feature Coefficients ----------------
st.markdown("""
<div class="card">
    <h3>Feature Coefficients</h3>
</div>
""", unsafe_allow_html=True)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

st.dataframe(coef_df)

# ==================================================
# 🔮 PREDICTION SECTION
# ==================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Smoker Status")

total_bill = st.slider(
    "Total Bill",
    float(df["total_bill"].min()),
    float(df["total_bill"].max()),
    25.0
)

size = st.slider(
    "Party Size",
    int(df["size"].min()),
    int(df["size"].max()),
    2
)

sex = st.selectbox("Sex", ["Male", "Female"])
day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])

# Build input dictionary
input_data = {
    "total_bill": total_bill,
    "size": size,
    "sex_Male": 1 if sex == "Male" else 0,
    "day_Fri": 1 if day == "Fri" else 0,
    "day_Sat": 1 if day == "Sat" else 0,
    "day_Sun": 1 if day == "Sun" else 0,
    "time_Dinner": 1 if time == "Dinner" else 0
}

input_df = pd.DataFrame([input_data])

# Add missing columns
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Align column order
input_df = input_df[X.columns]

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

result = "Smoker " if prediction == 1 else "Non-Smoker "

st.markdown(
    f'<div class="prediction-box">{result}<br>Probability: {probability:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
