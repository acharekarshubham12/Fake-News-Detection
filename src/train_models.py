# scripts/train_models.py
"""
Training scripts for all four models in the Fake News Detection Pipeline.
Saves trained models, vectorizers/tokenizers, and performance visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
import joblib
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from datasets import Dataset
import evaluate
from preprocessing import preprocess_text_basic, preprocess_text_advanced, preprocess_text_distilbert, extract_metadata, load_and_preprocess_data, balance_dataset

# Set up paths
DATA_PATH = "data/"
MODEL_PATH = "saved_models/"
IMG_PATH = "images/"
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(IMG_PATH, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name, filename):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name (str): Name of the model.
        filename (str): Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(IMG_PATH, filename))
    plt.close()

def train_model_1():
    """
    Train Model 1: TF-IDF Logistic Regression.
    """
    # Load and preprocess data
    df = load_and_preprocess_data(
        os.path.join(DATA_PATH, "True.csv"),
        os.path.join(DATA_PATH, "Fake.csv"),
        os.path.join(DATA_PATH, "clean_news.csv")
    )
    df = balance_dataset(df)
    
    # Split data
    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english', max_df=0.8, min_df=5)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Train Logistic Regression
    lr = LogisticRegression(class_weight='balanced', max_iter=3000, C=0.1)
    lr.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = lr.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = report['weighted avg']['f1-score']
    acc = accuracy_score(y_test, y_pred)
    print("Model 1 - TF-IDF Logistic Regression")
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, "TF-IDF Logistic Regression", "cm_tfidf_logistic.png")
    
    # Save model and vectorizer
    joblib.dump(lr, os.path.join(MODEL_PATH, "logistic_model.pkl"))
    joblib.dump(tfidf, os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))
    
    return f1, acc

def train_model_2():
    """
    Train Model 2: Random Forest with Metadata.
    """
    df = pd.read_csv(os.path.join(DATA_PATH, "clean_news.csv"))
    df = extract_metadata(pd.read_csv(os.path.join(DATA_PATH, "True.csv")).append(pd.read_csv(os.path.join(DATA_PATH, "Fake.csv"))))
    df['clean_text'] = df['text'].apply(preprocess_text_basic)
    df.dropna(subset=['clean_text'], inplace=True)
    
    # Split data
    X = df[['clean_text', 'title_len', 'body_len', 'title_body_ratio', 'exclamation_count', 'question_count', 'uppercase_ratio']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # TF-IDF and Metadata
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train['clean_text'].fillna(''))
    X_test_tfidf = tfidf.transform(X_test['clean_text'].fillna(''))
    X_train_meta = X_train[['title_len', 'body_len', 'title_body_ratio', 'exclamation_count', 'question_count', 'uppercase_ratio']].values
    X_test_meta = X_test[['title_len', 'body_len', 'title_body_ratio', 'exclamation_count', 'question_count', 'uppercase_ratio']].values
    X_train_combined = hstack([X_train_tfidf, X_train_meta])
    X_test_combined = hstack([X_test_tfidf, X_test_meta])
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_combined, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test_combined)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = report['weighted avg']['f1-score']
    acc = accuracy_score(y_test, y_pred)
    print("Model 2 - Random Forest with Metadata")
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, "Random Forest with Metadata", "cm_rf_metadata.png")
    
    # Save model and vectorizer
    joblib.dump(rf, os.path.join(MODEL_PATH, "rf_metadata.pkl"))
    joblib.dump(tfidf, os.path.join(MODEL_PATH, "tfidf_vectorizer_rf.pkl"))
    
    return f1, acc

def train_model_3():
    """
    Train Model 3: GloVe + BiGRU.
    """
    df = pd.read_csv(os.path.join(DATA_PATH, "clean_news.csv"))
    df.dropna(subset=['clean_text'], inplace=True)
    df['label'] = 1 - df['label']  # Flip labels
    df['processed_text'] = df['clean_text'].apply(preprocess_text_advanced)
    df = df[df['processed_text'].str.len() > 0]
    
    # Split data
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Tokenization
    vocab_size = 100000
    max_len = 100
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
    
    # Load GloVe embeddings
    embedding_index = {}
    glove_path = os.path.join(DATA_PATH, "glove.6B/glove.6B.100d.txt")
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vector
    
    embedding_dim = 100
    embedding_matrix = np.random.normal(size=(vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i >= vocab_size:
            continue
        vector = embedding_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
    
    # Build model
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False),
        Bidirectional(GRU(64, dropout=0.3)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    checkpoint = ModelCheckpoint(os.path.join(MODEL_PATH, "optimized_biGRU_model.keras"), monitor='val_accuracy', save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)
    model.fit(X_train_pad, y_train, validation_split=0.2, epochs=5, batch_size=256, callbacks=[checkpoint, early_stop, reduce_lr])
    
    # Evaluate
    y_pred_proba = model.predict(X_test_pad, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    print("Model 3 - GloVe + BiGRU")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    plot_confusion_matrix(y_test, y_pred, "GloVe + BiGRU", "cm_glove_bigru.png")
    
    # Save model and tokenizer
    model.save(os.path.join(MODEL_PATH, "optimized_biGRU_model.keras"))
    with open(os.path.join(MODEL_PATH, "tokenizer.pkl"), 'wb') as f:
        pickle.dump(tokenizer, f)
    
    return f1, acc

def train_model_4():
    """
    Train Model 4: DistilBERT Ensemble.
    """
    df_true = pd.read_csv(os.path.join(DATA_PATH, "True.csv"))
    df_fake = pd.read_csv(os.path.join(DATA_PATH, "Fake.csv"))
    df_true['combined_text'] = df_true.apply(lambda row: f"{row['title']}. {row['text']}", axis=1)
    df_fake['combined_text'] = df_fake.apply(lambda row: f"{row['title']}. {row['text']}", axis=1)
    df_true['label'] = 1
    df_fake['label'] = 0
    df = pd.concat([df_true[['combined_text', 'label']], df_fake[['combined_text', 'label']]]).sample(frac=1, random_state=42).reset_index(drop=True)
    df.rename(columns={'combined_text': 'text'}, inplace=True)
    df = df.sample(n=8000, random_state=42).reset_index(drop=True)
    df['text'] = df['text'].apply(preprocess_text_distilbert)
    df = df[df['text'].str.len() > 50].reset_index(drop=True)
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    # DistilBERT
    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=256, return_attention_mask=True)
    
    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)
    
    columns_to_remove = [col for col in train_dataset.column_names if col not in ['input_ids', 'attention_mask', 'label']]
    train_dataset = train_dataset.remove_columns(columns_to_remove).rename_column("label", "labels")
    val_dataset = val_dataset.remove_columns(columns_to_remove).rename_column("label", "labels")
    test_dataset = test_dataset.remove_columns(columns_to_remove).rename_column("label", "labels")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=256, pad_to_multiple_of=8)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, id2label={0: "Fake", 1: "True"}, label2id={"Fake": 0, "True": 1}
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    def compute_metrics(eval_pred):
        logits, labels =