import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

print("Loading dataset...")
df = pd.read_csv('data/5000_dataset_genai.csv')

X = df[['question_text', 'average_score', 'correct_rate', 'score_variance']]
y = df['difficulty_label']

print("Building pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=3000), 'question_text'),
        ('num', StandardScaler(), ['average_score', 'correct_rate', 'score_variance'])
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

print("Training model (this will take a few seconds)...")
pipeline.fit(X, y)

joblib.dump(pipeline, 'model.pkl')
print("Model retrained successfully and saved uniquely for this scikit-learn version!")
