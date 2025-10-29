import streamlit as st
import pandas as pd
import joblib
import random

# Load model and scaler
model = joblib.load("wine_quality_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit page config
st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="centered")

# ---- Custom CSS Styling ----
st.markdown("""
    <style>
    /* Make the entire background a light wine color */
    html, body, [class*="stAppViewContainer"], [class*="stApp"], [class*="main"], [data-testid="stAppViewContainer"] {
        background-color: #f7e6eb !important; /* soft rosé pink */
        color: #333;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Main container card */
    .block-container {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    /* Buttons */
    div.stButton > button {
        background-color: #b34766; /* wine pink */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 18px;
        font-size: 15px;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background-color: #d96685;
        transform: scale(1.05);
    }

    /* Input styling */
    .stNumberInput input {
        background-color: #fff;
        color: #000;
        border-radius: 6px;
        border: 1px solid #ccc;
    }

    /* Titles */
    h1 {
        color: #992e47;
        text-align: center;
        font-size: 2rem;
    }

    h3 {
        color: #7a1a35;
        font-size: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Result boxes */
    .good {
        background-color: #e8f5e9;
        color: #155724;
        border-left: 6px solid #28a745;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }

    .bad {
        background-color: #fdecea;
        color: #721c24;
        border-left: 6px solid #dc3545;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)



# ---- Title Section ----
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.title("🍷 Wine Quality Prediction App")
st.markdown("""
Predict whether a wine is **Good** or **Bad** based on its chemical properties.  
Use the **Next** or **Prev** buttons to switch between wines or adjust values manually.
""")

# ---- Features and Example Cases ----
features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

example_cases = [
    {'data': {'fixed acidity': 7.30, 'volatile acidity': 0.65, 'citric acid': 0.00, 'residual sugar': 1.20,
            'chlorides': 0.065, 'free sulfur dioxide': 15.0, 'total sulfur dioxide': 21.0,
            'density': 0.9946, 'pH': 3.39, 'sulphates': 0.47, 'alcohol': 10.0}},
    {'data': {'fixed acidity': 9.5, 'volatile acidity': 0.88, 'citric acid': 0.00, 'residual sugar': 2.3,
            'chlorides': 0.09, 'free sulfur dioxide': 5.0, 'total sulfur dioxide': 15.0,
            'density': 0.997, 'pH': 3.65, 'sulphates': 0.45, 'alcohol': 9.4}},
    {'data': {'fixed acidity': 7.8, 'volatile acidity': 0.58, 'citric acid': 0.02, 'residual sugar': 2.0,
            'chlorides': 0.073, 'free sulfur dioxide': 9.0, 'total sulfur dioxide': 18.0,
            'density': 0.9968, 'pH': 3.36, 'sulphates': 0.57, 'alcohol': 9.5}},
    {'data': {'fixed acidity': 7.7, 'volatile acidity': 0.69, 'citric acid': 0.22, 'residual sugar': 1.9,
            'chlorides': 0.084, 'free sulfur dioxide': 18.0, 'total sulfur dioxide': 94.0,
            'density': 0.9961, 'pH': 3.31, 'sulphates': 0.48, 'alcohol': 9.5}},
    {'data': {'fixed acidity': 8.8, 'volatile acidity': 0.61, 'citric acid': 0.30, 'residual sugar': 2.8,
            'chlorides': 0.088, 'free sulfur dioxide': 17.0, 'total sulfur dioxide': 46.0,
            'density': 0.9976, 'pH': 3.26, 'sulphates': 0.51, 'alcohol': 9.3}},
    {'data': {'fixed acidity': 8.5, 'volatile acidity': 0.28, 'citric acid': 0.56, 'residual sugar': 1.8,
            'chlorides': 0.092, 'free sulfur dioxide': 35.0, 'total sulfur dioxide': 103.0,
            'density': 0.9969, 'pH': 3.30, 'sulphates': 0.75, 'alcohol': 10.5}}
]

# ---- Navigation ----
if "case_index" not in st.session_state:
    st.session_state.case_index = 0

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("Previous"):
        st.session_state.case_index = (st.session_state.case_index - 1) % len(example_cases)
with col3:
    if st.button("Next"):
        st.session_state.case_index = (st.session_state.case_index + 1) % len(example_cases)

current_case = example_cases[st.session_state.case_index]

# ---- Input Section ----
st.markdown("##### Enter or Adjust Feature Values")

input_data = {}
for feature in features:
    input_data[feature] = st.number_input(
        f"{feature}:",
        min_value=0.0,
        value=float(current_case["data"][feature]),
        format="%.4f"
    )

input_df = pd.DataFrame([input_data])

# ---- Prediction Section ----
if st.button("🔮 Predict Quality"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.markdown("<p class='good'>✅ This wine is predicted to be <b>Good Quality</b>.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='bad'>❌ This wine is predicted to be <b>Bad Quality</b>.</p>", unsafe_allow_html=True)

    st.markdown("**Input Data Used:**")
    st.dataframe(input_df.style.format(precision=3))

st.markdown("</div>", unsafe_allow_html=True)
