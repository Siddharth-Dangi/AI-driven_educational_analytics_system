import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("data/5000_dataset_genai.csv")
# Features and target
X = df[["question_text", "average_score", "correct_rate", "score_variance"]]
y = df["difficulty_label"]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=3000), "question_text"),
        ("num", StandardScaler(), ["average_score", "correct_rate", "score_variance"])
    ]
)

# Pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

fig = plt.figure()
ax = fig.add_subplot(111)

cax = ax.matshow(cm)
plt.xticks(range(len(model.classes_)), model.classes_)
plt.yticks(range(len(model.classes_)), model.classes_)

for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, f"{val}", va='center', ha='center')

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.close()

# Save model
joblib.dump(model, "model.pkl")
