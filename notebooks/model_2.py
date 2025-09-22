from google.colab import drive
drive.mount('/content/drive')
import os, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
# Load cleaned dataset
dataset_path = "/content/drive/MyDrive/fakenews_project/clean_news.csv"
df = pd.read_csv(dataset_path)

# Drop rows with NaN in clean_text
df.dropna(subset=['clean_text'], inplace=True)

print("Dataset shape:", df.shape)
print(df.head())

#Feature Engineering (Metadata)
df['title_len'] = df['title'].apply(lambda x: len(str(x))) #Title Length
df['body_len'] = df['text'].apply(lambda x: len(str(x))) #Body Lenght
df['title_body_ratio'] = df['title_len'] / (df['body_len'] + 1)
df['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!')) #!
df['question_count'] = df['text'].apply(lambda x: str(x).count('?')) #?

#Upper case Ratio
def uppercase_ratio(text):
    words = str(text).split()
    if len(words) == 0:
        return 0
    upper_words = [w for w in words if w.isupper()]
    return len(upper_words) / len(words)
df['uppercase_ratio'] = df['text'].apply(uppercase_ratio)
print("✅ Metadata features added!")
print(df[['title_len','body_len','title_body_ratio',
          'exclamation_count','question_count','uppercase_ratio']].head())

# Define features and target
X = df[['clean_text', 'title_len','body_len','title_body_ratio',
        'exclamation_count','question_count','uppercase_ratio']]
y = df['label']

# Train-test split (after dropping NaNs)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


# Separate text and metadata features
X_train_text = X_train['clean_text'].fillna('')
X_test_text = X_test['clean_text'].fillna('')

X_train_meta = X_train[['title_len','body_len','title_body_ratio',
                        'exclamation_count','question_count','uppercase_ratio']].values
X_test_meta  = X_test[['title_len','body_len','title_body_ratio',
                       'exclamation_count','question_count','uppercase_ratio']].values


#TF-IDF
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words='english')


X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

print("TF-IDF shape:", X_train_tfidf.shape)
#combine the metadata with tf idf
X_train_combined = hstack([X_train_tfidf, X_train_meta])
X_test_combined  = hstack([X_test_tfidf,  X_test_meta])
print(" Combined feature shape:", X_train_combined.shape)

#ogistic Regression with metadata
lr2 = LogisticRegression(class_weight='balanced', max_iter=2000)
lr2.fit(X_train_combined, y_train)          #train
y_pred_lr2 = lr2.predict(X_test_combined)   #test

print("\n📌 Logistic Regression with Metadata:")
print(classification_report(y_test, y_pred_lr2))

#Random Forest with  metadata
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_combined, y_train)
y_pred_rf = rf.predict(X_test_combined)

print("\n Random Forest with Metadata:")
print(classification_report(y_test, y_pred_rf))


# F1 with Logistic Regression + Metadata
f1_lr2 = f1_score(y_test, y_pred_lr2, average='weighted')

# F1 with Random Forest + Metadata
f1_rf  = f1_score(y_test, y_pred_rf, average='weighted')

print("\n Model Comparison (Weighted F1):")

print(f"Logistic Regression + Metadata: {f1_lr2:.4f}")
print(f"Random Forest + Metadata: {f1_rf:.4f}")
save_path = "/content/drive/MyDrive/fakenews_project/saved_models/"
os.makedirs(save_path, exist_ok=True)
joblib.dump(rf, save_path + "rf_metadata.pkl")
joblib.dump(tfidf, save_path + "tfidf_vectorizer_rf.pkl")
print("✅ Random Forest model saved")