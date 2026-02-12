import streamlit as st
from electric import train_energy_model
from triage_model import train_triage_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Energy & Healthcare Dashboard",
    layout="wide"
)

st.title("🧠 AI-Based Energy & Healthcare Dashboard")

tab1, tab2 = st.tabs([
    "⚡ Renewable Energy Optimization",
    "🏥 Emergency Healthcare Triage"
])

# =======================
# PART A: ENERGY DASHBOARD
# =======================
with tab1:
    st.header("Renewable Energy Optimization")

    st.info(
        "Optimal Energy is calculated using the formula:\n\n"
        "Optimal Energy = 0.6 × Solar Power + 0.4 × Wind Power\n\n"
        "This represents a weighted contribution of renewable sources."
    )

    model, mse = train_energy_model()

    solar = st.slider("Solar Power (MW)", 0.0, 100.0, 50.0)
    wind = st.slider("Wind Power (MW)", 0.0, 100.0, 30.0)
    demand = st.slider("Electricity Demand (MW)", 0.0, 150.0, 90.0)

    prediction = model.predict([[solar, wind, demand]])

    col1, col2 = st.columns(2)
    col1.metric("Optimal Energy Output (MW)", round(prediction[0], 2))
    col2.metric("Model Error (MSE)", round(mse, 4))

    st.subheader("Demand Coverage")
    st.progress(min(prediction[0] / demand, 1.0))
    st.caption("This bar shows how much of the electricity demand is met by renewable energy")

    if prediction[0] >= demand:
        st.success("Renewable energy meets the electricity demand ✅")
    else:
        st.warning("Renewable energy is insufficient to meet demand ⚠️")

# ==========================
# PART B: HEALTHCARE DASHBOARD
# ==========================
with tab2:
    st.header("Emergency Healthcare Triage System")

    st.info(
        "This system uses Natural Language Processing (NLP) "
        "to analyze patient symptoms and predict urgency levels."
    )

    triage_model, vectorizer = train_triage_model()

    symptom_text = st.text_area(
        "Enter patient symptoms",
        "chest pain and dizziness"
    )

    if st.button("Predict Urgency"):
        st.markdown("### 🩺 Triage Decision Result")

        transformed = vectorizer.transform([symptom_text])
        urgency = triage_model.predict(transformed)[0]

        if urgency == 3:
            st.error("🚨 HIGH URGENCY – Immediate Attention Required")
        elif urgency == 2:
            st.warning("⚠️ MEDIUM URGENCY – Monitor Patient")
        else:
            st.success("✅ LOW URGENCY – Normal Care")

        st.caption(
            "Urgency prediction is performed using a TF-IDF based "
            "Naive Bayes classification model."
        )

