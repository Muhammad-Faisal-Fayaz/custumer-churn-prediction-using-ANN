from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
ONEHOT_ENCODER_PATH = BASE_DIR / "onehot_encoder.pkl"


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with SCALER_PATH.open("rb") as file:
        scaler = pickle.load(file)
    with LABEL_ENCODER_PATH.open("rb") as file:
        label_encoder = pickle.load(file)
    with ONEHOT_ENCODER_PATH.open("rb") as file:
        onehot_encoder = pickle.load(file)
    return model, scaler, label_encoder, onehot_encoder


def build_features(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_of_products,
    has_cr_card,
    is_active_member,
    estimated_salary,
    label_encoder,
    onehot_encoder,
):
    gender_encoded = label_encoder.transform([gender])[0]
    geography_encoded = onehot_encoder.transform([[geography]])
    geography_columns = list(onehot_encoder.get_feature_names_out(["Geography"]))

    features = pd.DataFrame(
        [[
            credit_score,
            gender_encoded,
            age,
            tenure,
            balance,
            num_of_products,
            has_cr_card,
            is_active_member,
            estimated_salary,
            *geography_encoded[0],
        ]],
        columns=[
            "CreditScore",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            *geography_columns,
        ],
    )

    # Keep the same feature order used by the notebook after one-hot encoding.
    return features[
        [
            "CreditScore",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Geography_France",
            "Geography_Germany",
            "Geography_Spain",
        ]
    ]


st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #f4f0e8;
        --muted: #a8aaa6;
        --panel: #171a1d;
        --panel-soft: #1e2225;
        --line: #30363a;
        --lime: #d9f36a;
    }

    .stApp {
        background: radial-gradient(circle at 82% 0%, #294243 0%, #111416 34%, #0c0e10 76%);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    .block-container { max-width: 1080px; padding: 3.5rem 2rem 4rem; }
    h1, h2, h3, .hero-title { font-family: 'Space Grotesk', sans-serif !important; }
    .hero { margin-bottom: 2.2rem; }
    .eyebrow { color: var(--lime); font-size: .76rem; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; }
    .hero-title { color: var(--ink); font-size: clamp(2.4rem, 6vw, 4.8rem); letter-spacing: -.05em; line-height: .98; margin: .65rem 0 .9rem; }
    .hero-copy { color: var(--muted); font-size: 1.05rem; max-width: 480px; line-height: 1.6; }
    .form-kicker { color: var(--lime); font-size: .72rem; font-weight: 700; letter-spacing: .15em; margin: .2rem 0 1.1rem; text-transform: uppercase; }
    [data-testid='stForm'] { background: rgba(23, 26, 29, .88); border: 1px solid var(--line); border-radius: 18px; padding: 1.8rem 1.8rem 1.45rem; box-shadow: 0 24px 60px rgba(0,0,0,.24); }
    [data-testid='stForm'] > div:first-child { gap: 2rem; }
    [data-testid='stWidgetLabel'] p { color: #d6d7d2; font-size: .82rem; font-weight: 600; }
    div[data-baseweb='input'], div[data-baseweb='select'] > div { background: var(--panel-soft); border: 1px solid var(--line); border-radius: 9px; }
    div[data-baseweb='input']:focus-within, div[data-baseweb='select'] > div:focus-within { border-color: var(--lime); box-shadow: 0 0 0 1px var(--lime); }
    input { color: #141719 !important; caret-color: #141719; }
    input::placeholder { color: #737a7d !important; opacity: 1; }
    [data-baseweb='select'] span { color: #141719; }
    [data-testid='stCheckbox'] { background: var(--panel-soft); border: 1px solid var(--line); border-radius: 9px; padding: .55rem .8rem; margin: .35rem 0 .7rem; }
    [data-testid='stCheckbox'] label p { color: #d6d7d2; }
    button[kind='primaryFormSubmit'] { background: var(--lime); border: 0; border-radius: 9px; color: #111416; font-weight: 700; min-height: 3rem; margin-top: 1rem; transition: transform .2s ease, box-shadow .2s ease; }
    button[kind='primaryFormSubmit']:hover { background: #e7ff82; box-shadow: 0 8px 24px rgba(217,243,106,.2); transform: translateY(-1px); }
    [data-testid='stMetric'] { background: linear-gradient(135deg, #252c2a, #171a1d); border: 1px solid #3e5144; border-radius: 16px; padding: 1.2rem 1.4rem; margin-top: .6rem; }
    [data-testid='stMetricLabel'] { color: var(--muted); }
    [data-testid='stMetricValue'] { color: var(--lime); font-family: 'Space Grotesk', sans-serif; }
    [data-testid='stProgressBar'] > div > div { background: var(--lime); }
    .result-heading { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; margin: 2.2rem 0 .3rem; }
    @media (max-width: 700px) { .block-container { padding: 2rem 1rem 3rem; } [data-testid='stForm'] { padding: 1.2rem; } .hero-title { font-size: 3rem; } }
    </style>
    <div class="hero">
        <div class="eyebrow">Retention intelligence / 01</div>
        <div class="hero-title">Customer<br>Churn Predictor</div>
        <div class="hero-copy">Turn customer signals into a clear retention decision.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, scaler, label_encoder, onehot_encoder = load_artifacts()
except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.stop()

with st.form("customer_details"):
    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown('<div class="form-kicker">Customer profile</div>', unsafe_allow_html=True)
        credit_score = st.number_input(
            "Credit score", min_value=300, max_value=900, value=None, placeholder="e.g. 650",
            help="Customer credit score, usually between 300 and 900.",
        )
        geography = st.selectbox(
            "Geography", ["France", "Germany", "Spain"], index=None,
            placeholder="Choose a country", help="Country associated with the customer account.",
        )
        gender = st.selectbox(
            "Gender", ["Female", "Male"], index=None,
            placeholder="Choose gender", help="Gender category used by the trained model.",
        )
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=None, placeholder="e.g. 40",
            help="Customer age in years.",
        )
        tenure = st.number_input(
            "Tenure (years)", min_value=0, max_value=10, value=None, placeholder="e.g. 3",
            help="How many years the customer has been with the bank.",
        )

    with right_column:
        st.markdown('<div class="form-kicker">Account signals</div>', unsafe_allow_html=True)
        balance = st.number_input(
            "Balance", min_value=0.0, value=None, step=1000.0, placeholder="e.g. 75,000",
            help="Current account balance in the dataset currency.",
        )
        num_of_products = st.number_input(
            "Number of products", min_value=1, max_value=4, value=None, placeholder="e.g. 1",
            help="Number of bank products held by the customer.",
        )
        has_cr_card = st.checkbox("Has credit card", value=False, help="Whether the customer has a credit card.")
        is_active_member = st.checkbox("Is active member", value=False, help="Whether the customer is an active member.")
        estimated_salary = st.number_input(
            "Estimated salary", min_value=0.0, value=None, step=1000.0, placeholder="e.g. 100,000",
            help="Estimated annual salary in the dataset currency.",
        )

    submitted = st.form_submit_button("Predict churn", type="primary")

if submitted:
    required_values = {
        "Credit score": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "Number of products": num_of_products,
        "Estimated salary": estimated_salary,
    }
    missing_values = [label for label, value in required_values.items() if value is None]
    if missing_values:
        st.warning("Complete the highlighted inputs before running a prediction.")
        st.stop()

    raw_features = build_features(
        credit_score,
        geography,
        gender,
        age,
        tenure,
        balance,
        num_of_products,
        int(has_cr_card),
        int(is_active_member),
        estimated_salary,
        label_encoder,
        onehot_encoder,
    )
    scaled_features = scaler.transform(raw_features)
    churn_probability = float(model.predict(scaled_features, verbose=0)[0][0])

    st.markdown('<div class="result-heading">Prediction result</div>', unsafe_allow_html=True)
    st.metric("Churn probability", f"{churn_probability:.1%}")
    st.progress(churn_probability)

    chart_data = pd.DataFrame(
        {"Probability": [1 - churn_probability, churn_probability]},
        index=["Likely to stay", "Likely to churn"],
    )
    st.markdown("#### Outcome probability", unsafe_allow_html=True)
    st.bar_chart(chart_data, horizontal=True, height=180, color="#d9f36a")

    if churn_probability >= 0.5:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is unlikely to churn.")
