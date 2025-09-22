# scripts/preprocessing.py
"""
Preprocessing functions for the Fake News Detection Pipeline.
Includes text cleaning, tokenization, and metadata extraction for all models.
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.utils import resample

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize stop words and stemmer/lemmatizer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text_basic(text):
    """
    Basic text preprocessing for Models 1 and 2 (stemming, stopword removal).
    
    Args:
        text (str): Input text to preprocess.
    
    Returns:
        str: Cleaned and stemmed text.
    """
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    text = ' '.join(ps.stem(word) for word in text.split() if word not in stop_words)
    return text

def preprocess_text_advanced(text):
    """
    Advanced text preprocessing for Model 3 (lemmatization, URL removal).
    
    Args:
        text (str): Input text to preprocess.
    
    Returns:
        str: Cleaned and lemmatized text.
    """
    if isinstance(text, float) or text is None:
        return ''
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s\.\?\!]', ' ', text)  # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
    return text

def preprocess_text_distilbert(text):
    """
    Text preprocessing for Model 4 (DistilBERT).
    
    Args:
        text (str): Input text to preprocess.
    
    Returns:
        str: Cleaned text for transformer model.
    """
    if pd.isna(text) or text == "":
        return ""
    text = str(text).lower()
    text = re.sub(r'\b(reuters|associated press|ap|—)\b', '', text)
    text = re.sub(r'\([^)]*\)\s*-?\s*', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    return text.strip()

def extract_metadata(df):
    """
    Extract metadata features for Model 2 (title length, body length, etc.).
    
    Args:
        df (pd.DataFrame): DataFrame with 'title' and 'text' columns.
    
    Returns:
        pd.DataFrame: DataFrame with added metadata columns.
    """
    df['title_len'] = df['title'].apply(lambda x: len(str(x)))
    df['body_len'] = df['text'].apply(lambda x: len(str(x)))
    df['title_body_ratio'] = df['title_len'] / (df['body_len'] + 1)
    df['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!'))
    df['question_count'] = df['text'].apply(lambda x: str(x).count('?'))
    df['uppercase_ratio'] = df['text'].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()]) / len(str(x).split()) if len(str(x).split()) > 0 else 0
    )
    return df

def load_and_preprocess_data(true_path, fake_path, output_path):
    """
    Load and preprocess the dataset, combining true and fake news.
    
    Args:
        true_path (str): Path to True.csv.
        fake_path (str): Path to Fake.csv.
        output_path (str): Path to save preprocessed dataset.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with 'clean_text' and 'label' columns.
    """
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
    
    # Add labels
    df_true['label'] = 1
    df_fake['label'] = 0
    
    # Combine datasets
    df = pd.concat([df_true, df_fake])
    
    # Apply basic preprocessing for Models 1 and 2
    df['clean_text'] = df['text'].apply(preprocess_text_basic)
    df.dropna(subset=['clean_text'], inplace=True)
    
    # Save preprocessed dataset
    df[['clean_text', 'label']].to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to {output_path}")
    
    return df

def balance_dataset(df):
    """
    Balance the dataset by upsampling the minority class.
    
    Args:
        df (pd.DataFrame): DataFrame with 'clean_text' and 'label' columns.
    
    Returns:
        pd.DataFrame: Balanced DataFrame.
    """
    df_fake = df[df['label'] == 0]
    df_true = df[df['label'] == 1]
    df_true_upsampled = resample(df_true, replace=True, n_samples=len(df_fake), random_state=42)
    df_balanced = pd.concat([df_fake, df_true_upsampled]).sample(frac=1, random_state=42)
    print("Balanced Dataset shape:", df_balanced.shape)
    print(df_balanced['label'].value_counts())
    return df_balanced