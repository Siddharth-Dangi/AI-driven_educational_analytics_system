import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

st.set_page_config(page_title="Exam Question Analytics", layout="wide")

# --- Color palette ---
COLORS = {"Easy": "#1f8b4c", "Medium": "#d4a017", "Hard": "#b02a37"}
PALETTE = list(COLORS.values())

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

st.title("Exam Question Analytics")
st.markdown("Advanced ML based Difficulty Prediction & Assessment Intelligence")

tabs = st.tabs(["Batch Analytics", "Single Prediction"])

# =====================================================
# TAB 1 — BATCH ANALYTICS
# =====================================================
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

            # ---------- KPI Metrics ----------
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Questions", len(uploaded_df))
            col2.metric("Average Confidence", f"{uploaded_df['Confidence_%'].mean():.2f}%")
            col3.metric("Unique Classes", uploaded_df["Predicted_Difficulty"].nunique())

            st.markdown("### Preview")
            st.dataframe(uploaded_df.head(10), use_container_width=True)

            # ---------- 1. Difficulty Distribution (Pie + Bar) ----------
            st.markdown("---")
            st.markdown("### 📊 Predicted Difficulty Distribution")
            diff_counts = uploaded_df["Predicted_Difficulty"].value_counts()
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig1, ax1 = plt.subplots(figsize=(5, 5))
                color_list = [COLORS.get(label, "#888") for label in diff_counts.index]
                ax1.pie(diff_counts, labels=diff_counts.index, autopct="%1.1f%%",
                        colors=color_list, startangle=140,
                        textprops={"fontsize": 12, "color": "white"})
                ax1.set_title("Difficulty Split", color="white", fontsize=14)
                fig1.patch.set_facecolor("#0e1117")
                st.pyplot(fig1)

            with chart_col2:
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                bars = ax2.bar(diff_counts.index, diff_counts.values,
                               color=color_list, edgecolor="white", linewidth=0.5)
                for bar, val in zip(bars, diff_counts.values):
                    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                             str(val), ha="center", va="bottom", color="white", fontsize=12)
                ax2.set_ylabel("Count", color="white")
                ax2.set_title("Question Count by Difficulty", color="white", fontsize=14)
                ax2.tick_params(colors="white")
                ax2.set_facecolor("#1c1f26")
                fig2.patch.set_facecolor("#0e1117")
                st.pyplot(fig2)

            # ---------- 2. Confidence Distribution Histogram ----------
            st.markdown("---")
            st.markdown("### 📈 Confidence Score Distribution")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            for label in ["Easy", "Medium", "Hard"]:
                subset = uploaded_df[uploaded_df["Predicted_Difficulty"] == label]["Confidence_%"]
                if not subset.empty:
                    ax3.hist(subset, bins=20, alpha=0.65, label=label,
                             color=COLORS.get(label, "#888"), edgecolor="white", linewidth=0.5)
            ax3.set_xlabel("Confidence %", color="white")
            ax3.set_ylabel("Frequency", color="white")
            ax3.set_title("Confidence Distribution by Predicted Difficulty", color="white", fontsize=14)
            ax3.legend()
            ax3.tick_params(colors="white")
            ax3.set_facecolor("#1c1f26")
            fig3.patch.set_facecolor("#0e1117")
            st.pyplot(fig3)

            # ---------- 3. Feature Averages by Difficulty (Grouped Bar) ----------
            st.markdown("---")
            st.markdown("### 📉 Feature Averages by Predicted Difficulty")
            feature_cols = ["average_score", "correct_rate", "score_variance"]
            group_means = uploaded_df.groupby("Predicted_Difficulty")[feature_cols].mean()
            # Reorder to Easy, Medium, Hard
            order = [l for l in ["Easy", "Medium", "Hard"] if l in group_means.index]
            group_means = group_means.loc[order]

            fig4, ax4 = plt.subplots(figsize=(10, 5))
            x = np.arange(len(order))
            width = 0.25
            for i, feat in enumerate(feature_cols):
                bars = ax4.bar(x + i * width, group_means[feat], width, label=feat,
                               edgecolor="white", linewidth=0.5)
                for bar in bars:
                    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                             f"{bar.get_height():.2f}", ha="center", va="bottom",
                             color="white", fontsize=9)
            ax4.set_xticks(x + width)
            ax4.set_xticklabels(order)
            ax4.set_ylabel("Mean Value", color="white")
            ax4.set_title("Avg Score · Correct Rate · Variance  per Difficulty", color="white", fontsize=14)
            ax4.legend()
            ax4.tick_params(colors="white")
            ax4.set_facecolor("#1c1f26")
            fig4.patch.set_facecolor("#0e1117")
            st.pyplot(fig4)

            # ---------- 4. Scatter Plot — Score vs Correct Rate ----------
            st.markdown("---")
            st.markdown("### 🔍 Score vs Correct Rate (colored by Difficulty)")
            fig5, ax5 = plt.subplots(figsize=(10, 5))
            for label in ["Easy", "Medium", "Hard"]:
                subset = uploaded_df[uploaded_df["Predicted_Difficulty"] == label]
                if not subset.empty:
                    ax5.scatter(subset["average_score"], subset["correct_rate"],
                                alpha=0.5, label=label, color=COLORS.get(label, "#888"),
                                edgecolors="white", linewidth=0.3, s=30)
            ax5.set_xlabel("Average Score", color="white")
            ax5.set_ylabel("Correct Rate", color="white")
            ax5.set_title("Average Score vs Correct Rate", color="white", fontsize=14)
            ax5.legend()
            ax5.tick_params(colors="white")
            ax5.set_facecolor("#1c1f26")
            fig5.patch.set_facecolor("#0e1117")
            st.pyplot(fig5)

            # ---------- 5. Box Plot — Score Variance by Difficulty ----------
            st.markdown("---")
            st.markdown("### 📦 Score Variance Distribution by Difficulty")
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            box_data = [uploaded_df[uploaded_df["Predicted_Difficulty"] == l]["score_variance"].dropna()
                        for l in order]
            bp = ax6.boxplot(box_data, labels=order, patch_artist=True,
                             boxprops=dict(linewidth=1.2),
                             medianprops=dict(color="white", linewidth=2))
            for patch, label in zip(bp["boxes"], order):
                patch.set_facecolor(COLORS.get(label, "#888"))
            ax6.set_ylabel("Score Variance", color="white")
            ax6.set_title("Score Variance Spread per Difficulty", color="white", fontsize=14)
            ax6.tick_params(colors="white")
            ax6.set_facecolor("#1c1f26")
            fig6.patch.set_facecolor("#0e1117")
            st.pyplot(fig6)

            # ---------- 6. Topic-wise Difficulty Breakdown ----------
            if "topic" in uploaded_df.columns:
                st.markdown("---")
                st.markdown("### 🏷️ Topic-wise Difficulty Breakdown")
                topic_diff = pd.crosstab(uploaded_df["topic"], uploaded_df["Predicted_Difficulty"])
                # Reorder columns
                topic_diff = topic_diff[[c for c in ["Easy", "Medium", "Hard"] if c in topic_diff.columns]]
                fig7, ax7 = plt.subplots(figsize=(12, max(5, len(topic_diff) * 0.35)))
                topic_diff.plot(kind="barh", stacked=True, ax=ax7,
                                color=[COLORS.get(c, "#888") for c in topic_diff.columns],
                                edgecolor="white", linewidth=0.3)
                ax7.set_xlabel("Number of Questions", color="white")
                ax7.set_ylabel("Topic", color="white")
                ax7.set_title("Stacked Difficulty by Topic", color="white", fontsize=14)
                ax7.legend(title="Difficulty")
                ax7.tick_params(colors="white")
                ax7.set_facecolor("#1c1f26")
                fig7.patch.set_facecolor("#0e1117")
                st.pyplot(fig7)

            # ---------- Confusion Matrix (if ground truth exists) ----------
            if "difficulty_label" in uploaded_df.columns:
                st.markdown("---")
                st.markdown("### 🎯 Confusion Matrix")
                cm = confusion_matrix(uploaded_df["difficulty_label"], predictions, labels=model.classes_)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_,
                            yticklabels=model.classes_, cmap="Blues", ax=ax_cm,
                            linewidths=0.5, linecolor="#0e1117")
                ax_cm.set_xlabel("Predicted", color="white")
                ax_cm.set_ylabel("Actual", color="white")
                ax_cm.set_title("Confusion Matrix", color="white", fontsize=14)
                ax_cm.tick_params(colors="white")
                fig_cm.patch.set_facecolor("#0e1117")
                st.pyplot(fig_cm)

            # ---------- Download ----------
            st.markdown("---")
            csv = uploaded_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv,
                file_name="predicted_difficulty.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.error("CSV must contain: question_text, average_score, correct_rate, score_variance")

# =====================================================
# TAB 2 — SINGLE PREDICTION
# =====================================================
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
            proba_all = model.predict_proba(input_df)[0] * 100
            probability = np.max(proba_all)

            # Badge
            if prediction == "Easy":
                st.markdown(f"<div class='badge-easy'>{prediction}</div>", unsafe_allow_html=True)
            elif prediction == "Medium":
                st.markdown(f"<div class='badge-medium'>{prediction}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='badge-hard'>{prediction}</div>", unsafe_allow_html=True)

            st.progress(int(probability))
            st.write(f"Confidence: {probability:.2f}%")

            # --- Per-class probability bar chart ---
            st.markdown("#### Class Probabilities")
            fig_p, ax_p = plt.subplots(figsize=(5, 3))
            classes = model.classes_
            bar_colors = [COLORS.get(c, "#888") for c in classes]
            bars = ax_p.barh(classes, proba_all, color=bar_colors,
                             edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, proba_all):
                ax_p.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                          f"{val:.1f}%", va="center", color="white", fontsize=11)
            ax_p.set_xlim(0, 110)
            ax_p.set_xlabel("Probability %", color="white")
            ax_p.set_title("Prediction Probability per Class", color="white", fontsize=13)
            ax_p.tick_params(colors="white")
            ax_p.set_facecolor("#1c1f26")
            fig_p.patch.set_facecolor("#0e1117")
            st.pyplot(fig_p)