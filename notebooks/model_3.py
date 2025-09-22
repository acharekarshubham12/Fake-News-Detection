# ===============================
# Optimized Fake News Detection Model
# ===============================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import libraries
import os, re, numpy as np, pandas as pd, pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ====== Step 1: Load Dataset ======
dataset_path = "/content/drive/MyDrive/fakenews_project/clean_news.csv"
df = pd.read_csv(dataset_path)

# Drop rows with NaN in clean_text
df.dropna(subset=['clean_text'], inplace=True)

print("Dataset shape:", df.shape)
print("Label distribution:")
print(df['label'].value_counts())

# Check if we need to flip labels (based on previous analysis)
print("\nChecking label interpretation...")
sample_real = df[df['label'] == 1].iloc[0]['clean_text'][:200] if 1 in df['label'].values else ""
sample_fake = df[df['label'] == 0].iloc[0]['clean_text'][:200] if 0 in df['label'].values else ""

print("Sample REAL news (label 1):", sample_real)
print("Sample FAKE news (label 0):", sample_fake)

# Based on previous analysis, it appears labels need to be flipped
print("\nFlipping labels to correct interpretation (0=Fake, 1=Real)")
df['label'] = 1 - df['label']  # Flip labels: 0 becomes 1, 1 becomes 0
print("New label distribution:")
print(df['label'].value_counts())

# ====== Step 2: Enhanced Text Preprocessing ======
def preprocess_text(text):
    if isinstance(text, float) or text is None:
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z\s\.\?\!]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply preprocessing
print("Preprocessing text...")
df['processed_text'] = df['clean_text'].apply(preprocess_text)

# Check for empty texts after preprocessing
original_count = len(df)
df = df[df['processed_text'].str.len() > 0]
print(f"Removed {original_count - len(df)} rows with empty text after preprocessing")

# ====== Step 3: Train-Test Split ======
X = df['processed_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Class distribution in training: {np.bincount(y_train)}")
print(f"Class distribution in test: {np.bincount(y_test)}")

# ====== Step 4: Tokenize text and pad sequences ======
vocab_size = 100000
max_len = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Calculate optimal sequence length
train_lengths = [len(seq) for seq in X_train_seq]
print(f"Average sequence length: {np.mean(train_lengths):.2f}")
print(f"95th percentile length: {np.percentile(train_lengths, 95):.2f}")

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# ====== Step 5: Load GloVe embeddings ======
embedding_index = {}
try:
    glove_path = "/content/drive/MyDrive/fakenews_project/glove.6B/glove.6B.100d.txt"
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vector
    print(f"✅ GloVe embeddings loaded with {len(embedding_index)} word vectors!")
except FileNotFoundError:
    print("❌ GloVe embeddings file not found. Using random embeddings.")
    use_glove = False
else:
    use_glove = True

# ====== Step 6: Create embedding matrix ======
embedding_dim = 100
embedding_matrix = np.random.normal(size=(vocab_size, embedding_dim))

if use_glove:
    found = 0
    for word, i in tokenizer.word_index.items():
        if i >= vocab_size:
            continue
        vector = embedding_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
            found += 1

    print(f"Found embeddings for {found} words out of {vocab_size}")
    print(f"Coverage: {found/vocab_size*100:.2f}%")

# ====== Step 7: Build Enhanced Model ======
# Calculate class weights to handle imbalance
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")

model = Sequential()
model.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=max_len,
    trainable=False
))

model.add(Bidirectional(GRU(64, dropout=0.3)))  # Single layer, fewer units
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))






# Build the model properly
model.build(input_shape=(None, max_len))

# Use a lower learning rate for better stability
optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("Model Summary:")
model.summary()

# ====== Step 8: Callbacks ======
checkpoint = ModelCheckpoint(
    "optimized_biGRU_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.0001,
    verbose=1
)

# ====== Step 9: Train model ======
print("Training model...")
history = model.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=256,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# ====== Step 10: Evaluate model ======
model = load_model("optimized_biGRU_model.keras")

print("Evaluating on test data...")
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_accuracy:.4f}")
print(f"✅ Test Precision: {test_precision:.4f}")
print(f"✅ Test Recall: {test_recall:.4f}")

# Predictions
y_pred_proba = model.predict(X_test_pad, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ====== Step 11: Save model and tokenizer ======
save_path = "/content/drive/MyDrive/fakenews_project/"
os.makedirs(save_path, exist_ok=True)

model.save(os.path.join(save_path, "optimized_biGRU_model.keras"))
print("✅ Model saved to Google Drive!")

with open(os.path.join(save_path, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved to Google Drive!")

# ====== Step 12: Test with examples ======
def predict_news(text):
    """Predict if a news article is real or fake"""
    # Preprocess
    processed_text = preprocess_text(text)

    if len(processed_text) < 10:  # Too short to make prediction
        return "Unknown", 0.5, 0.5

    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # Predict
    prediction = model.predict(padded, verbose=0)
    probability = prediction[0][0]

    # Correct interpretation (after label flipping)
    label = "Real" if probability > 0.5 else "Fake"
    confidence = probability if label == "Real" else 1 - probability

    return label, confidence, probability

# Test examples
test_examples = [
    # Real news examples
    "Scientists have discovered a new species of dinosaur in Argentina that provides insight into the evolution of the giant sauropods.",
    "A new study shows that regular exercise can significantly improve cognitive function in older adults.",
    "Researchers have developed a new battery technology that could double the range of electric vehicles.",

    # Fake news examples
    "The government is implanting microchips in COVID vaccines to track everyone's movements and control their thoughts.",
    "Celebrities are secretly lizard people who control the world through mind control technology.",
    "NASA has confirmed that a massive asteroid will hit Earth next month, ending all life on the planet."
]

print("\n" + "="*50)
print("TESTING WITH EXAMPLE NEWS HEADLINES")
print("="*50)

for i, example in enumerate(test_examples, 1):
    label, confidence, prob = predict_news(example)
    print(f"\nExample {i}:")
    print(f"Text: {example}")
    print(f"Prediction: {label} (Confidence: {confidence:.2%})")
    print(f"Probability: {prob:.4f}")

# ====== Step 13: Test with samples from the dataset ======
print("\n" + "="*50)
print("TESTING WITH SAMPLES FROM THE DATASET")
print("="*50)

# Get some real samples from the dataset
real_samples = df[df['label'] == 1].head(2)
for i, (idx, row) in enumerate(real_samples.iterrows(), 1):
    label, confidence, prob = predict_news(row['clean_text'])
    print(f"\nReal Sample {i}:")
    print(f"Text: {row['clean_text'][:100]}...")
    print(f"True Label: Real (1)")
    print(f"Prediction: {label} (Confidence: {confidence:.2%})")
    print(f"Probability: {prob:.4f}")

# Get some fake samples from the dataset
fake_samples = df[df['label'] == 0].head(2)
for i, (idx, row) in enumerate(fake_samples.iterrows(), 1):
    label, confidence, prob = predict_news(row['clean_text'])
    print(f"\nFake Sample {i}:")
    print(f"Text: {row['clean_text'][:100]}...")
    print(f"True Label: Fake (0)")
    print(f"Prediction: {label} (Confidence: {confidence:.2%})")
    print(f"Probability: {prob:.4f}")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETE!")
print("="*50)
print("Model saved to:", os.path.join(save_path, "optimized_biGRU_model.keras"))
print("Tokenizer saved to:", os.path.join(save_path, "tokenizer.pkl"))