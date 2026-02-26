import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
st.set_page_config(page_title="Exam Question Analytics", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: white;
}
.metric-box {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
}
.badge-easy {
    background-color: #1f8b4c;
    padding: 8px 16px;
    border-radius: 20px;
    color: white;
    font-weight: bold;
}
.badge-medium {
    background-color: #d4a017;
    padding: 8px 16px;
    border-radius: 20px;
    color: white;
    font-weight: bold;
}
.badge-hard {
    background-color: #b02a37;
    padding: 8px 16px;
    border-radius: 20px;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("model.pkl")

st.title("Exam Question Analytics")
st.markdown("Advanced ML based Difficulty Prediction & Assessment Intelligence")

tabs = st.tabs(["Batch Analytics", "Single Prediction"])

with tabs[0]:
    st.subheader("Upload Dataset")
    st.markdown("""
    **Required CSV Format:**
    Your uploaded file must contain the following columns:
    - `question_text`
    - `average_score`
    - `correct_rate`
    - `score_variance`

    *(Optional)*: If `difficulty_label` column is included, the system will generate a Confusion Matrix.
    """)
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="batch")

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        required_columns = {"question_text", "average_score", "correct_rate", "score_variance"}
        if required_columns.issubset(uploaded_df.columns):
            predictions = model.predict(uploaded_df)
            probabilities = model.predict_proba(uploaded_df)

            uploaded_df["Predicted_Difficulty"] = predictions
            uploaded_df["Confidence_%"] = np.max(probabilities, axis=1) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Questions", len(uploaded_df))
            col2.metric("Average Confidence", f"{uploaded_df['Confidence_%'].mean():.2f}%")
            col3.metric("Unique Classes", uploaded_df["Predicted_Difficulty"].nunique())

            st.markdown("### Preview")
            st.dataframe(uploaded_df.head(), use_container_width=True)

            if "difficulty_label" in uploaded_df.columns:
                st.markdown("### Confusion Matrix")
                cm = confusion_matrix(uploaded_df["difficulty_label"], predictions, labels=model.classes_)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            csv = uploaded_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="predicted_difficulty.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.error("CSV must contain: question_text, average_score, correct_rate, score_variance")

with tabs[1]:
    st.subheader("Single Question Analysis")

    colA, colB = st.columns(2)

    with colA:
        question = st.text_area("Question Text")
        avg_score = st.number_input("Average Score (0-10)", 0.0, 10.0)
        correct_rate = st.number_input("Correct Rate (0-1)", 0.0, 1.0)
        variance = st.number_input("Score Variance", 0.0)

    with colB:
        if st.button("Run Analysis", use_container_width=True):
            input_df = pd.DataFrame([{
                "question_text": question,
                "average_score": avg_score,
                "correct_rate": correct_rate,
                "score_variance": variance
            }])
            prediction = model.predict(input_df)[0]
            probability = np.max(model.predict_proba(input_df)) * 100
            
            if prediction == "Easy":
                st.markdown(f"<div class='badge-easy'>{prediction}</div>", unsafe_allow_html=True)
            elif prediction == "Medium":
                st.markdown(f"<div class='badge-medium'>{prediction}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='badge-hard'>{prediction}</div>", unsafe_allow_html=True)

            st.progress(int(probability))
            st.write(f"Confidence: {probability:.2f}%")