import streamlit as st
import pandas as pd
import joblib
import json


# -----------------------------
# Load model and preprocessor
# -----------------------------
model = joblib.load("car_price_random_forest.pkl")
preprocessor = joblib.load("car_price_preprocessor.pkl")

with open("company_car_mapping.json", "r") as f:
    company_car_mapping = json.load(f)

# Get categories learned during training
categorical_encoder = preprocessor.named_transformers_["cat"]

name_options = list(categorical_encoder.categories_[0])
company_options = list(categorical_encoder.categories_[1])
fuel_options = list(categorical_encoder.categories_[2])


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)


# -----------------------------
# Custom styling
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .price-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 26px;
        font-weight: 700;
        margin-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">🚗 Used Car Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning powered used-car price estimation</div>',
    unsafe_allow_html=True
)

st.divider()


# -----------------------------
# Input section
# -----------------------------
st.subheader("Enter Car Details")


company = st.selectbox(
    "Company",
    company_options
)

company_cars = company_car_mapping.get(company, [])

name = st.selectbox(
    "Car Name",
    company_cars
)
year = st.number_input(
    "Manufacturing Year",
    min_value=1995,
    max_value=2019,
    value=2017,
    step=1
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=500000,
    value=30000,
    step=1000
)

fuel_type = st.selectbox(
    "Fuel Type",
    fuel_options
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Price", use_container_width=True):

    new_car = pd.DataFrame({
        "name": [name],
        "company": [company],
        "year": [year],
        "kms_driven": [kms_driven],
        "fuel_type": [fuel_type]
    })

    try:
        # Preprocess input
        new_car_processed = preprocessor.transform(new_car)

        # Prediction
        predicted_price = model.predict(
            new_car_processed
        )[0]

        st.success("✅ Prediction completed successfully!")

        st.markdown("## 💰 Estimated Car Price")

        st.markdown(
            f"# ₹{predicted_price:,.0f}"
        )

        st.caption(
            "Estimated using a Random Forest regression model."
        )

        st.info(
             "Note: This is an ML-based estimate and may differ from the actual market price."
        )
    except Exception:
        st.error(
            "Prediction could not be completed. "
            "Please try another combination of car details."
        )
        