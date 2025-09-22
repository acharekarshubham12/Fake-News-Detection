#TF-IDF
# ================================
# Fake News Detection - Diagnostic Code with Gradio
# ================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import libraries
import pandas as pd
import os, joblib
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import gradio as gr
nltk.download('stopwords')

# ------------------------
# Preprocessing function
# ------------------------
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    text = ' '.join(ps.stem(word) for word in text.split() if word not in stop_words)
    return text

# ------------------------
# Load and preprocess dataset
# ------------------------
true_path = "/content/drive/MyDrive/fakenews_project/True.csv"
fake_path = "/content/drive/MyDrive/fakenews_project/Fake.csv"
df_true = pd.read_csv(true_path)
df_fake = pd.read_csv(fake_path)

# Add labels
df_true['label'] = 1
df_fake['label'] = 0

# Combine and preprocess
df = pd.concat([df_true, df_fake])
df['clean_text'] = df['text'].apply(preprocess_text)
df.dropna(subset=['clean_text'], inplace=True)

print("Original Dataset shape:", df.shape)
print(df['label'].value_counts())

# Save preprocessed dataset
output_path = "/content/drive/MyDrive/fakenews_project/clean_news.csv"
df[['clean_text', 'label']].to_csv(output_path, index=False)
print(f"Preprocessed dataset saved to {output_path}")

# ------------------------
# Handle class imbalance
# ------------------------
df_fake = df[df['label'] == 0]
df_true = df[df['label'] == 1]

df_true_upsampled = resample(df_true,
                             replace=True,
                             n_samples=len(df_fake),
                             random_state=42)

df_balanced = pd.concat([df_fake, df_true_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42)

print("Balanced Dataset shape:", df_balanced.shape)
print(df_balanced['label'].value_counts())

# ------------------------
# Split features and labels
# ------------------------
X = df_balanced['clean_text']
y = df_balanced['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------
# TF-IDF Vectorization
# ------------------------
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english',
    max_df=0.8,
    min_df=5
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF shape:", X_train_tfidf.shape)

# ------------------------
# Train Logistic Regression
# ------------------------
lr = LogisticRegression(class_weight='balanced', max_iter=3000, C=0.1)
lr.fit(X_train_tfidf, y_train)

# ------------------------
# Evaluate on test set
# ------------------------
y_pred = lr.predict(X_test_tfidf)
report = classification_report(y_test, y_pred, output_dict=True)
baseline_f1 = report['weighted avg']['f1-score']

print("\nClassification Report on Test Set:\n")
print(classification_report(y_test, y_pred))
print(f"Baseline F1 Score: {baseline_f1:.4f}")

# ------------------------
# Feature Importance
# ------------------------
feature_names = tfidf.get_feature_names_out()
coef = lr.coef_[0]
top_fake = sorted(zip(coef, feature_names), reverse=False)[:10]
top_true = sorted(zip(coef, feature_names), reverse=True)[:10]
print("\nTop features for Fake:", top_fake)
print("Top features for True:", top_true)

# ------------------------
# Test on provided articles
# ------------------------
true_samples = [
    "HARRISONBURG, Va. (Reuters) - U.S. President Donald Trump returns to the site of his first campaign rally on Saturday, seeking to rally his supporters in the wake of a week of political turmoil and falling approval ratings...",
    "WASHINGTON (Reuters) - A top Federal Reserve official said on Tuesday that the U.S. central bank does not need to rush into raising interest rates, with inflation still running below target and the job market near full employment...",
    "BAGHDAD (Reuters) - Iraqi forces backed by U.S.-led coalition air strikes retook a village north of Mosul from Islamic State on Tuesday, a spokesman said, as part of a wider offensive to drive the militants from their de facto capital...",
    "SEOUL (Reuters) - Samsung Electronics Co Ltd said on Friday its second-quarter profit likely halved from a year earlier, as weak demand for memory chips and smartphones offset gains from its display business...",
    "LONDON (Reuters) - British Prime Minister Theresa May's Conservatives were on course to win Thursday's election but lose their majority in parliament, exit polls showed, a result that would leave her needing support from another party to govern..."
]

# Sample fake articles (from ISOT Fake.csv examples)
fake_samples = [
    "Hillary Clinton Caught in Massive Scandal! Emails Prove She Sold State Secrets to Aliens!",
    "BREAKING: Obama Admits He Was Born in Kenya, Plans to Flee Country!",
    "Scientists Discover Vaccines Cause Autism, Big Pharma in Panic!",
    "Trump Declares Martial Law, Claims Election Was Stolen by Lizard People!",
    "Elvis Presley Found Alive in Secret Government Bunker!"
]

sample_texts = true_samples + fake_samples
sample_tfidf = tfidf.transform([preprocess_text(text) for text in sample_texts])
sample_preds = lr.predict(sample_tfidf)
sample_probs = lr.predict_proba(sample_tfidf)

print("\nSample Predictions with Probabilities:\n")
for text, pred, prob in zip(sample_texts, sample_preds, sample_probs):
    label = "Fake" if pred == 0 else "True"
    print(f"Predicted: {label}  |  Probabilities -> Fake: {prob[0]:.3f}, True: {prob[1]:.3f}")
    print(f"News: {text[:100]}...\n")

# ------------------------
# Save model and vectorizer
# ------------------------
save_path = "/content/drive/MyDrive/fakenews_project/saved_models/"
os.makedirs(save_path, exist_ok=True)
joblib.dump(lr, os.path.join(save_path, "logistic_model.pkl"))
joblib.dump(tfidf, os.path.join(save_path, "tfidf_vectorizer.pkl"))
print("✅ Model and vectorizer saved successfully!")

