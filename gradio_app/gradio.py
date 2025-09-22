import gradio as gr
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import re
from scipy.special import softmax

# Load the saved models
def load_models():
    """Load both transformer and content-based models"""
    model_path = "/content/drive/MyDrive/fakenews_project/saved_models/"
    
    # Load transformer model
    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        model_path + "distilbert_model"
    )
    transformer_tokenizer = AutoTokenizer.from_pretrained(
        model_path + "distilbert_model"
    )
    
    # Load content-based model
    with open(model_path + "content_model.pkl", "rb") as f:
        content_model = pickle.load(f)
    
    return transformer_model, transformer_tokenizer, content_model

# Load models
transformer_model, transformer_tokenizer, content_model = load_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer_model.to(device)

def preprocess_text(text):
    """Basic text cleaning"""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def transformer_predict(texts):
    """Get predictions from transformer model"""
    inputs = transformer_tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
        predicted_classes = torch.argmax(predictions, dim=-1)
    
    return predictions.cpu().numpy(), predicted_classes.cpu().numpy()

def content_predict(texts):
    """Get predictions from content-based model"""
    if isinstance(texts, str):
        texts = [texts]
    
    predictions = content_model.predict(texts)
    probabilities = content_model.predict_proba(texts)
    
    return probabilities, predictions

def ensemble_predict(text):
    """Combine predictions from both models"""
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Get transformer predictions
    trans_probs, trans_preds = transformer_predict([cleaned_text])
    trans_prob = trans_probs[0]
    trans_pred = trans_preds[0]
    
    # Get content model predictions
    cont_probs, cont_preds = content_predict(cleaned_text)
    cont_prob = cont_probs[0]
    cont_pred = cont_preds[0]
    
    # Weighted average (70% content, 30% transformer)
    combined_prob = 0.7 * cont_prob + 0.3 * trans_prob
    final_pred = np.argmax(combined_prob)
    final_confidence = combined_prob[final_pred] * 100
    
    # Get individual model confidences
    trans_confidence = trans_prob[trans_pred] * 100
    cont_confidence = cont_prob[cont_pred] * 100
    
    # Prepare results
    results = {
        "ensemble": {
            "prediction": "REAL" if final_pred == 1 else "FAKE",
            "confidence": final_confidence
        },
        "transformer": {
            "prediction": "REAL" if trans_pred == 1 else "FAKE",
            "confidence": trans_confidence
        },
        "content": {
            "prediction": "REAL" if cont_pred == 1 else "FAKE",
            "confidence": cont_confidence
        },
        "probabilities": {
            "real_prob": combined_prob[1] * 100,
            "fake_prob": combined_prob[0] * 100
        }
    }
    
    return results

def create_visualization(results):
    """Create visualization of the results"""
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Model confidence comparison
    models = ['Transformer', 'Content', 'Ensemble']
    confidences = [
        results['transformer']['confidence'],
        results['content']['confidence'], 
        results['ensemble']['confidence']
    ]
    colors = ['lightblue', 'lightgreen', 'gold']
    
    bars = ax1.bar(models, confidences, color=colors, edgecolor='black')
    ax1.set_ylabel('Confidence (%)')
    ax1.set_title('Model Confidence Comparison')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, confidence in zip(bars, confidences):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{confidence:.1f}%', ha='center', va='bottom')
    
    # Probability distribution
    labels = ['FAKE', 'REAL']
    probabilities = [results['probabilities']['fake_prob'], 
                    results['probabilities']['real_prob']]
    colors = ['red' if results['ensemble']['prediction'] == 'FAKE' else 'lightcoral', 
             'green' if results['ensemble']['prediction'] == 'REAL' else 'lightgreen']
    
    wedges, texts, autotexts = ax2.pie(probabilities, labels=labels, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title('Probability Distribution')
    
    plt.tight_layout()
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_str}"

def analyze_news(text):
    """Main function to analyze news text"""
    if not text or len(text.strip()) < 20:
        return "Please enter a longer news article for analysis.", None, None
    
    try:
        # Get predictions
        results = ensemble_predict(text)
        
        # Create visualization
        visualization = create_visualization(results)
        
        # Prepare detailed results
        detailed_results = f"""
        ## 📊 Analysis Results
        
        ### Ensemble Prediction: **{results['ensemble']['prediction']}** ({results['ensemble']['confidence']:.1f}% confidence)
        
        ### Individual Model Results:
        - 🤖 **Transformer Model**: {results['transformer']['prediction']} ({results['transformer']['confidence']:.1f}% confidence)
        - 📝 **Content Model**: {results['content']['prediction']} ({results['content']['confidence']:.1f}% confidence)
        
        ### Probability Distribution:
        - 🔴 FAKE: {results['probabilities']['fake_prob']:.1f}%
        - 🟢 REAL: {results['probabilities']['real_prob']:.1f}%
        """
        
        # Additional insights based on prediction
        if results['ensemble']['prediction'] == 'FAKE':
            insight = "🔍 **Insight**: This article shows characteristics commonly associated with fake news, such as sensational language, lack of credible sources, or conspiracy theories."
        else:
            insight = "🔍 **Insight**: This article appears credible with factual reporting, proper sourcing, and neutral language typical of real news."
        
        detailed_results += f"\n\n{insight}"
        
        return detailed_results, visualization, results['ensemble']['prediction']
    
    except Exception as e:
        return f"Error analyzing text: {str(e)}", None, None

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Fake News Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🕵️‍♂️ Fake News Detection System")
        gr.Markdown("This tool uses advanced AI to analyze news articles and determine their credibility.")
        
        with gr.Row():
            with gr.Column(scale=2):
                news_input = gr.Textbox(
                    label="Paste News Article Text",
                    placeholder="Enter the full text of the news article you want to analyze...",
                    lines=10,
                    max_lines=20
                )
                
                analyze_btn = gr.Button("Analyze Article", variant="primary")
            
            with gr.Column(scale=1):
                result_output = gr.Markdown(label="Analysis Results")
                image_output = gr.HTML(label="Visualization")
                final_verdict = gr.Textbox(label="Final Verdict", interactive=False)
        
        # Examples
        gr.Markdown("### 📋 Example Articles")
        examples = [
            ["Scientists confirm that regular exercise significantly improves cardiovascular health and reduces disease risk."],
            ["New study shows vaccination effectively prevents severe illness and reduces transmission rates."],
            ["Alien technology is being suppressed to maintain fossil fuel industry profits."]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=news_input,
            outputs=[result_output, image_output, final_verdict],
            fn=analyze_news,
            cache_examples=False
        )
        
        analyze_btn.click(
            fn=analyze_news,
            inputs=news_input,
            outputs=[result_output, image_output, final_verdict]
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
      
    