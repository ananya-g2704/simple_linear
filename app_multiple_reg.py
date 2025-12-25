import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------ Page Config ------------------
st.set_page_config("Multiple Linear Regression", layout="centered")

# ------------------ Load CSS ------------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style_multiple.css")

# ------------------ Title ------------------
st.markdown("""
<div class="card">
    <h1>Multiple Linear Regression</h1>
    <p>Predict <b>Tip Amount</b> using multiple features from the <b>Tips Dataset</b>.</p>
</div>
""", unsafe_allow_html=True)

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# ------------------ Dataset Preview ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Data Preparation ------------------
# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("tip", axis=1)
y = df_encoded["tip"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ------------------ Train Model ------------------
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# ------------------ Metrics ------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)

# ------------------ Visualization ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Actual vs Predicted Tip Amount")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red", linestyle="--")
ax.set_xlabel("Actual Tip")
ax.set_ylabel("Predicted Tip")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Performance ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adj R²", f"{adj_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Coefficients ------------------
st.markdown("""
<div class="card">
    <h3>Model Coefficients</h3>
</div>
""", unsafe_allow_html=True)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

st.dataframe(coef_df)

# ------------------ Prediction Section ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

total_bill = st.slider(
    "Total Bill",
    float(df["total_bill"].min()),
    float(df["total_bill"].max()),
    30.0
)

size = st.slider(
    "Party Size",
    int(df["size"].min()),
    int(df["size"].max()),
    2
)


sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])

# Build input row
input_data = {
    'total_bill': total_bill,
    'size': size,
    'sex_Male': 1 if sex == "Male" else 0,
    'smoker_Yes': 1 if smoker == "Yes" else 0,
    'day_Fri': 1 if day == "Fri" else 0,
    'day_Sat': 1 if day == "Sat" else 0,
    'day_Sun': 1 if day == "Sun" else 0,
    'time_Dinner': 1 if time == "Dinner" else 0
}

input_df = pd.DataFrame([input_data])

# Add missing columns if any
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[X.columns]
input_scaled = scaler.transform(input_df)
predicted_tip = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box"> Predicted Tip : $ {predicted_tip:.2f} </div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
