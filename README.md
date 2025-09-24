# 📰 Fake News Detection Pipeline

This repository contains a **complete fake news detection pipeline** built with **machine learning** and **deep learning** techniques.  
The system classifies news articles as **Fake** or **True** by leveraging both **content-based models** and **transformer-based models**, and further enhances accuracy using an **ensemble approach**.

---

## 🚀 Features
- End-to-end pipeline: data cleaning, preprocessing, training, evaluation, and saving models.
- Combines **traditional ML** and **state-of-the-art NLP** methods.
- Supports **custom realistic example testing** to evaluate generalization.
- Implements **ensemble learning** for robust predictions.
- Saves trained models for easy deployment.

---

## 📂 Dataset
- **Source:** Kaggle Fake News Dataset (`True.csv` and `Fake.csv`)
- Preprocessing:
  - Combined `title` + `text` for richer context.
  - Applied regex-based cleaning (removed sources, links, special characters).
  - Filtered short/low-quality samples.
- Balanced dataset sampled for efficiency.

---
## Flow Diagram
<img width="1536" height="1024" alt="ChatGPT Image Sep 24, 2025, 07_49_59 PM" src="https://github.com/user-attachments/assets/321a834a-ee4f-45a3-9a28-bb57f0f8be07" />

---
## 🧠 Models Implemented

### **Model_1 – Logistic Regression with TF-IDF**
- Approach: Classic ML baseline using **TF-IDF (1–2 grams)** features.  
- Handles text content only.  
- Strong baseline with **balanced class weights**.  

---

### **Model_2 – DistilBERT Transformer**
- Approach: Fine-tuned **DistilBERT (base-uncased)** for sequence classification.  
- Optimized with dropout, warmup, and weighted F1 evaluation.  
- Trained with **Hugging Face Transformers + Trainer API**.  

---

### **Model_3 – Content Model with Style-Neutral Augmentation**
- Approach: Re-trained TF-IDF + Logistic Regression with additional **style-neutral examples**.  
- Improves **generalization** and reduces overfitting to dataset-specific writing styles.  

---

### **Model_4 – Ensemble (Content + Transformer)**
- Approach: Combines Model_2 and Model_3 predictions.  
- **Weighted averaging (70% content, 30% transformer)** for robustness.  
- Achieves best performance on **realistic unseen examples**.  

---

## 📊 Evaluation

### Metrics
- **Accuracy**
- **Weighted F1 Score**
- **Classification Report**
- **Confusion Matrix**

### Results Overview
| Model         | Test Accuracy | Custom Example Accuracy |
|---------------|--------------:|------------------------:|
| Model_1 (TF-IDF + Logistic Regression) | Good baseline | Moderate generalization |
| Model_2 (DistilBERT Transformer) | High accuracy | Sensitive to style shifts |
| Model_3 (TF-IDF + Augmentation) | Improved robustness | Better generalization |
| Model_4 (Ensemble) | Best overall | Strongest performance |

---

## 💾 Model Saving
All trained models are saved under `saved_models/` for deployment:
- **Model_1:** `content_model.pkl` (TF-IDF + Logistic Regression)  
- **Model_2:** `distilbert_model/` (Transformer + Tokenizer)  
- **Model_3:** `content_model.pkl` (augmented retrain)  
- **Model_4:** Ensemble logic implemented in pipeline  

Usage example:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle

# Load Transformer
transformer_model = AutoModelForSequenceClassification.from_pretrained("saved_models/distilbert_model")
transformer_tokenizer = AutoTokenizer.from_pretrained("saved_models/distilbert_model")

# Load Content Model
with open("saved_models/content_model.pkl", "rb") as f:
    content_model = pickle.load(f)

```
## 🏗️ Tech Stack

- Python
- Hugging Face Transformers
- Scikit-learn
- Panas / NumPy
- Matplotlib
- Torch

## 🎯 Key Highlights
- Built an end-to-end NLP pipeline handling preprocessing → training → evaluation → saving.
- Integrated traditional ML & deep learning approaches.
- Improved real-world robustness via augmentation + ensemble.
- Production-ready with saved models for deployment.


## ROC Curve Comparision
<img width="1022" height="691" alt="Roc Curve Comparison" src="https://github.com/user-attachments/assets/36bee5e9-dff6-4865-a5c7-f9a4f7cdf53e" />

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests with detailed descriptions of changes. Ensure all tests pass and maintain the existing code style.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or support, please contact [acharekarshubham12@gmail.com](mailto:acharekarshubham12@gmail.com).

