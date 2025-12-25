import streamlit as st 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------- Page Config ---------------- #
st.set_page_config("Ridge & Lasso Regression", layout="centered")


# ---------------- Load CSS ---------------- #
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style_ridge_lasso.css")


# ---------------- Load Data ---------------- #
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()


# ---------------- Title ---------------- #
st.markdown("""
<div class="card">
<h1>Ridge & Lasso Regression</h1>
<p>Predict <b>Tip</b> from <b>Total Bill</b> using Ridge and Lasso Regression</p>
</div>
""", unsafe_allow_html=True)


# ---------------- Dataset Preview ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Prepare Data ---------------- #
X = df[['total_bill']]
y = df['tip']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------- Model Selection ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Select Regression Model")

model_choice = st.radio(
    "Choose model",
    ("Ridge Regression", "Lasso Regression")
)

alpha = st.slider(
    "Regularization Strength (alpha)",
    0.01, 5.0, 1.0
)

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Train Selected Model ---------------- #
if model_choice == "Ridge Regression":
    model = Ridge(alpha=alpha)
else:
    model = Lasso(alpha=alpha)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


# ---------------- Metrics ---------------- #
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)


# ---------------- Visualization ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"Total Bill vs Tip ({model_choice})")

fig, ax = plt.subplots()
ax.scatter(df['total_bill'], df['tip'], alpha=0.6, label="Actual Data")

bill_range = np.array(df['total_bill']).reshape(-1, 1)
bill_scaled = scaler.transform(bill_range)

ax.plot(
    df['total_bill'],
    model.predict(bill_scaled),
    color='red',
    linewidth=2.5,
    label=model_choice
)


ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
ax.legend()
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Performance ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"{model_choice} Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adj R²", f"{adj_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Coefficient & Intercept ---------------- #
st.markdown(f"""
<div class="card">
<h3>{model_choice} – Model Parameters</h3>

<b>Coefficient:</b> {model.coef_[0]:.3f}<br>
<b>Intercept:</b> {model.intercept_:.3f}

</div>
""", unsafe_allow_html=True)


# ---------------- Prediction ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"Predict Tip Amount ({model_choice})")

bill = st.slider(
    "Total Bill",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

bill_scaled = scaler.transform([[bill]])
prediction = model.predict(bill_scaled)[0]

st.markdown(
    f"""
    <div class="prediction-box">
    Predicted Tip: $ {prediction:.2f}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
