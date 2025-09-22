# scripts/predict.py
"""
Prediction utilities for the Fake News Detection Pipeline.
Includes functions for predicting with all models and a Gradio interface for Model 1.
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocessing import preprocess_text_basic, preprocess_text_advanced, preprocess_text_distilbert

# Set up paths
MODEL_PATH = "saved_models/"

def load_model_1():
    """
    Load Model 1 (TF-IDF Logistic Regression) and its vectorizer.
    
    Returns:
        tuple: Trained model and TF-IDF vectorizer.
    """
    model = joblib.load(os.path.join(MODEL_PATH, "logistic_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))
    return model, vectorizer

def load_model_2():
    """
    Load Model 2 (Random Forest with Metadata) and its vectorizer.
    
    Returns:
        tuple: Trained model and TF-IDF vectorizer.
    """
    model = joblib.load(os.path.join(MODEL_PATH, "rf_metadata.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_PATH, "tfidf_vectorizer_rf.pkl"))
    return model, vectorizer

def load_model_3():
    """
    Load Model 3 (GloVe + BiGRU) and its tokenizer.
    
    Returns:
        tuple: Trained model and tokenizer.
    """
    model = load_model(os.path.join(MODEL_PATH, "optimized_biGRU_model.keras"))
    with open(os.path.join(MODEL_PATH, "tokenizer.pkl"), 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def load_model_4():
    """
    Load Model 4 (DistilBERT Ensemble) and its components.
    
    Returns:
        tuple: DistilBERT model, tokenizer, and content model.
    """
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_PATH, "distilbert_model"))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, "distilbert_model"))
    content_model = joblib.load(os.path.join(MODEL_PATH, "content_model.pkl"))
    return model, tokenizer, content_model

def predict_model_1(text, model, vectorizer):
    """
    Predict with Model 1.
    
    Args:
        text (str): Input text.
        model: Trained logistic regression model.
        vectorizer: TF-IDF vectorizer.
    
    Returns:
        tuple: Label and probability.
    """
    clean_text = preprocess_text_basic(text)
    text_tfidf = vectorizer.transform([clean_text])
    prob = model.predict_proba(text_tfidf)[0]
    label = "Fake" if model.predict(text_tfidf)[0] == 0 else "True"
    return label, prob[1] if label == "True" else prob[0]

def predict_model_2(text, title, model, vectorizer):
    """
    Predict with Model 2.
    
    Args:
        text (str): Input text.
        title (str): Input title.
        model: Trained random forest model.
        vectorizer: TF-IDF vectorizer.
    
    Returns:
        tuple: Label and probability.
    """
    clean_text = preprocess_text_basic(text)
    df = pd.DataFrame({'title': [title], 'text': [text]})
    df = extract_metadata(df)
    text_tfidf = vectorizer.transform([clean_text])
    meta_features = df[['title_len', 'body_len', 'title_body_ratio', 'exclamation_count', 'question_count', 'uppercase_ratio']].values
    combined_features = hstack([text_tfidf, meta_features])
    prob = model.predict_proba(combined_features)[0]
    label = "Fake" if model.predict(combined_features)[0] == 0 else "True"
    return label, prob[1] if label == "True" else prob[0]

def predict_model_3(text, model, tokenizer):
    """
    Predict with Model 3.
    
    Args:
        text (str): Input text.
        model: Trained BiGRU model.
        tokenizer: Keras tokenizer.
    
    Returns:
        tuple: Label and probability.
    """
    clean_text = preprocess_text_advanced(text)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    prob = model.predict(padded, verbose=0)[0][0]
    label = "Real" if prob > 0.5 else "Fake"
    confidence = prob if label == "Real" else 1 - prob
    return label, confidence

def predict_model_4(text, trans_model, trans_tokenizer, content_model):
    """
    Predict with Model 4 (ensemble).
    
    Args:
        text (str): Input text.
        trans_model: DistilBERT model.
        trans_tokenizer: DistilBERT tokenizer.
        content_model: TF-IDF logistic regression model.
    
    Returns:
        tuple: Label and confidence.
    """
    clean_text = preprocess_text_distilbert(text)
    # DistilBERT prediction
    inputs = trans_tokenizer(clean_text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = trans_model(**inputs)
    trans_probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
    # Content model prediction
    content_probs = content_model.predict_proba([clean_text])[0]
    # Ensemble: 70% content, 30% transformer
    combined_prob = 0.7 * content_probs + 0.3 * trans_probs
    pred = np.argmax(combined_prob)
    label = "Fake" if pred == 0 else "True"
    confidence = combined_prob[pred]
    return label, confidence

def gradio_interface(text):
    """
    Gradio interface for Model 1 predictions.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Prediction result with confidence