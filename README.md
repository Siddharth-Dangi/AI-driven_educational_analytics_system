# AI-Driven Educational Analytics System

### Intelligent Exam Question Difficulty Prediction using Feature Engineering & Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Supported-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)

[Overview](#-overview) · [Architecture](#-system-architecture) · [Quickstart](#-quickstart) · [Model](#-model) · [Results](#-results) · [Application](#-application) · [Visualizations](#-visualizations) · [Limitations](#-limitations) · [Future Work](#-future-work)

---

## 🎯 Overview

Educators and testing organizations spend significant effort manually evaluating the difficulty of examination questions. Misjudged difficulty levels can skew assessment outcomes and inaccurately measure student proficiency.

The **AI-Driven Educational Analytics System** is an end-to-end ML pipeline that automatically predicts question difficulty. Given a question's text and post-exam statistics (average score, correct rate, score variance), it classifies the question into one of three difficulty tiers:

| Tier | Description |
|------|-------------|
| 🟢 **Easy** | High average score, high correct rate |
| 🟡 **Medium** | Moderate score, moderate correct rate |
| 🔴 **Hard** | Low average score, low correct rate |

### Problem Statement

> Given a dataset of 5,000 exam questions with associated student performance metrics, build a machine learning model that can accurately predict question difficulty levels and deploy it as an interactive web application for real-time analysis.

### Key Features

- **Batch Analytics** — Upload a CSV of questions and get instant difficulty predictions for all of them
- **Single Question Prediction** — Predict difficulty for an individual question with confidence scores
- **Rich Visualizations** — 7 interactive charts in the web app + 14 model performance plots in notebook
- **Confusion Matrix & Classification Report** — Detailed per-class performance analysis
- **Downloadable Results** — Export predictions as CSV

---

## 🏗 System Architecture

The system operates across three main components:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│  5000_dataset_genai.csv                                              │
│  ┌────────────┬──────────────┬──────────────┬────────────────┐       │
│  │ question_  │ average_     │ correct_     │ score_         │       │
│  │ text       │ score        │ rate         │ variance       │       │
│  └────────────┴──────────────┴──────────────┴────────────────┘       │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     MODEL PIPELINE                                   │
│                                                                      │
│  ┌──────────────┐    ┌────────────────┐    ┌─────────────────┐       │
│  │ TF-IDF       │    │ Standard       │    │ Logistic        │       │
│  │ Vectorizer   │───▶│ Scaler         │───▶│ Regression      │       │
│  │ (3000 feat.) │    │ (Numerical)    │    │ (max_iter=1000) │       │
│  └──────────────┘    └────────────────┘    └─────────────────┘       │
│         ColumnTransformer                      Classifier            │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER (Streamlit)                       │
│                                                                      │
│  ┌─────────────────────┐    ┌──────────────────────────┐             │
│  │ Batch Analytics     │    │ Single Question          │             │
│  │ • Upload CSV        │    │ • Enter question text    │             │
│  │ • 7 visualizations  │    │ • Input scores           │             │
│  │ • Download results  │    │ • Get prediction + chart │             │
│  └─────────────────────┘    └──────────────────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
AI-driven_educational_analytics_system/
├── README.md                          # ← You are here
│
├── app.py                             # Streamlit web application
├── model.pkl                          # Serialised trained model (Pipeline)
├── confusion_matrix.png               # Generated confusion matrix plot
│
├── data/
│   └── 5000_dataset_genai.csv         # Dataset (5,000 exam questions)
│
└── notebook/
    ├── train.ipynb                    # Model training notebook
    └── visualizations.ipynb           # 14 model performance visualizations
```

---

## 🚀 Quickstart

### 1. Clone & enter the repo

```bash
git clone <your-repo-link>
cd AI-driven_educational_analytics_system
```

### 2. Set up environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** If there is no `requirements.txt`, install dependencies manually:
> ```bash
> pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
> ```

### 3. Train the model

Open `notebook/train.ipynb` and run all cells. This will:
- Load the dataset from `data/5000_dataset_genai.csv`
- Train a Logistic Regression pipeline (TF-IDF + StandardScaler)
- Save the trained model as `model.pkl`

### 4. Launch the app

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### 5. Explore model performance (optional)

Open `notebook/visualizations.ipynb` and run all cells to generate 14 detailed performance charts.

---

## 🧠 Model

### Pipeline Overview

A single **scikit-learn Pipeline** combining text and numerical features:

| Stage | Component | Details |
|-------|-----------|---------|
| **Text Features** | `TfidfVectorizer` | Converts `question_text` into 3,000 TF-IDF features |
| **Numerical Features** | `StandardScaler` | Normalizes `average_score`, `correct_rate`, `score_variance` |
| **Feature Fusion** | `ColumnTransformer` | Combines text + numerical feature vectors |
| **Classifier** | `LogisticRegression` | Multi-class classification with `max_iter=1000` |

### Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | 5,000 exam questions |
| Train/Test Split | 80% / 20% (stratified) |
| Text Features | 3,000 (TF-IDF) |
| Numerical Features | 3 (average_score, correct_rate, score_variance) |
| Random State | 42 |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.90% |
| **Precision** (weighted) | 98.90% |
| **Recall** (weighted) | 98.90% |
| **F1-Score** (weighted) | 98.90% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 🟢 Easy | 0.99 | 1.00 | 0.99 | 334 |
| 🟡 Medium | 0.99 | 0.98 | 0.98 | 333 |
| 🔴 Hard | 0.99 | 0.99 | 0.99 | 333 |

---

## 🕹️ Application

The Streamlit app (`app.py`) provides **two tabs**:

### 1. Batch Analytics

Upload a CSV file containing exam questions and instantly receive:

- **KPI Metrics** — Total questions, average confidence, unique classes
- **Difficulty Distribution** — Pie chart + bar chart
- **Confidence Histogram** — Distribution by predicted class
- **Feature Averages** — Grouped bar chart of score, correct rate, variance per difficulty
- **Scatter Plot** — Average score vs correct rate, colored by difficulty
- **Box Plot** — Score variance distribution per difficulty
- **Topic Breakdown** — Stacked horizontal bar chart (if `topic` column exists)
- **Confusion Matrix** — If `difficulty_label` ground truth column is present
- **Download** — Export results as CSV

### 2. Single Question Prediction

Enter a question's text and metrics to get:

- Predicted difficulty with colored badge (Easy / Medium / Hard)
- Confidence progress bar
- Per-class probability bar chart

---

## 📈 Visualizations

The `notebook/visualizations.ipynb` notebook contains **14 detailed performance visualizations**:

| # | Visualization | Purpose |
|---|---------------|---------|
| 1 | Overall Metrics Bar Chart | Accuracy, Precision, Recall, F1 at a glance |
| 2 | Confusion Matrix (Counts) | Raw classification results |
| 3 | Confusion Matrix (Normalized) | Percentage-based view |
| 4 | Per-Class Precision/Recall/F1 | Grouped bar comparison |
| 5 | ROC Curves (One-vs-Rest) | AUC per class |
| 6 | Precision-Recall Curves | Average Precision per class |
| 7 | Confidence Distribution (Overall) | How confident is the model? |
| 8 | Confidence by Predicted Class | Per-class confidence spread |
| 9 | Correct vs Misclassified Confidence | Does confidence correlate with correctness? |
| 10 | Class Distribution (Train vs Test) | Verifies stratified split balance |
| 11 | Feature Violin Plots | Score, correct rate, variance by difficulty |
| 12 | Learning Curve | Does the model benefit from more data? |
| 13 | 5-Fold Cross-Validation | Per-fold accuracy consistency |
| 14 | Misclassification Analysis | Which classes get confused and where |

---

## 📋 Dataset

The dataset (`data/5000_dataset_genai.csv`) contains **5,000 exam questions** with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `question_id` | int | Unique identifier |
| `question_text` | string | Full text of the exam question |
| `topic` | string | Subject area (Chemistry, Data Structures, Biology, etc.) |
| `average_score` | float | Mean student score (0–10) |
| `correct_rate` | float | Proportion of correct responses (0–1) |
| `score_variance` | float | Variance in student scores |
| `difficulty_label` | string | Ground truth: Easy, Medium, or Hard |

---

## 🚦 Limitations

- **Language:** Optimized for English-language questions only
- **Post-Exam Features:** The model requires post-exam statistics (average score, correct rate, variance), which are only available after students have taken the exam
- **No Visual Understanding:** Cannot interpret images, diagrams, or graphs embedded in questions
- **Single Model:** Only Logistic Regression is implemented; ensemble methods may improve performance further

---

## 🚀 Future Work

- **Pre-Exam Model** — Build a text-only model using advanced NLP (BERT, sentence transformers) for difficulty prediction before administering the exam
- **LLM Integration** — Use Gemini/GPT to analyze question complexity through step-by-step reasoning
- **Multi-Modal Support** — Vision-Language models for diagram-dependent questions
- **Ensemble Methods** — Compare XGBoost, Random Forest, and SVM against the current Logistic Regression baseline
- **RAG for Curriculum Alignment** — Retrieval-Augmented Generation against educational standards
- **Real-Time Feedback Loops** — Continuously update the model as new exam data arrives

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core language |
| scikit-learn | ML pipeline, preprocessing, evaluation |
| Streamlit | Interactive web application |
| Matplotlib & Seaborn | Visualizations |
| Pandas & NumPy | Data manipulation |
| Joblib | Model serialization |

---

<p align="center">
  <b>Built with ❤️ for smarter educational assessments</b>
</p>
