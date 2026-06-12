import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')


st.title("❤️ Heart Disease Risk Predictor")
st.caption("Educational ML Project • Not Medical Advice")

st.markdown("""
<style>

/* Predict Button */
.stButton > button {

    background: linear-gradient(
        90deg,
        #ff4b6e,
        #ff6b6b
    );

    color: white;

    border-radius: 12px;

    height: 55px;

    border: none;

    font-size: 18px;

    font-weight: 600;

    transition: 0.3s;
}


/* Hover */
.stButton > button:hover {

    transform: scale(1.02);

    box-shadow:
        0 0 20px
        rgba(255,75,110,0.5);

}


/* Click */
.stButton > button:active {

    transform: scale(0.98);

}

</style>
""", unsafe_allow_html=True)

st.sidebar.title("About")

st.sidebar.info("""
❤️ Heart Disease Risk Predictor

🧠 Model:
K-Nearest Neighbors

🖥 Frontend:
Streamlit

☁️ Deployment:
Streamlit Cloud

⚠ Educational use only.
""")

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age",18,100,40)

    sex = st.selectbox(
        "Sex",
        ['M','F']
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ['ATA','NAP','TA','ASY']
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [0,1]
    )

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["Y","N"]
    )


with col2:

    resting_bp = st.number_input(
        "Resting BP",
        80,200,120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        100,600,200
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ['Normal','ST','LVH']
    )

    max_hr = st.slider(
        "Max Heart Rate",
        60,220,150
    )

    oldpeak = st.slider(
        "Oldpeak",
        0.0,6.0,1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ['Up','Flat','Down']
    )


if st.button(
    "🔍 Predict Risk",
    use_container_width=True
):
    raw_input = {
    'Age': age,
    'Sex_' + sex: 1,

    'ChestPainType_' + chest_pain: 1,

    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,

    'FastingBS': fasting_bs,

    'RestingECG_' + resting_ecg: 1,

    'MaxHR': max_hr,

    'ExerciseAngina_' + exercise_angina: 1,

    'Oldpeak': oldpeak,

    'ST_Slope_' + st_slope: 1
}

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]


    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.divider()
    st.subheader("📊 Prediction Result")

    confidence = None

    if hasattr(model, "predict_proba"):

        confidence = (
            model.predict_proba(
                scaled_input
            )[0][prediction]
        )

        st.metric(
            "Confidence",
            f"{confidence:.1%}"
        )

        st.progress(float(confidence))

    if prediction == 1:
        st.error(
            "⚠️ Higher likelihood of Heart Disease"
        )

    else:
        st.success(
            "✅ Lower likelihood of Heart Disease"
        )

st.divider()

st.caption(
    "Built with ❤️ using Python • Streamlit • Scikit-Learn"
)


