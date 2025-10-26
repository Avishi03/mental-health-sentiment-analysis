"""
Mental Health Sentiment Analysis - Gradio Web Application
Author: Avishi Agrawal
Date: October 2024
"""

import gradio as gr
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
MODEL_PATH = "mental_health_model"  # Adjust path as needed

try:
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f" Model loaded successfully on {device}")
except Exception as e:
    print(f" Error loading model: {e}")
    print("Using demo mode with mock predictions")
    MODEL_PATH = None

# Label names
LABEL_NAMES = ['Stress/Anxiety', 'Neutral/Normal']

def predict_sentiment(text):
    """
    Predict sentiment for input text
    
    Args:
        text: Input text string
        
    Returns:
        Formatted string with prediction results
    """
    if not text or not text.strip():
        return " Please enter some text to analyze."
    
    # Demo mode if model not loaded
    if MODEL_PATH is None:
        if any(word in text.lower() for word in ['stress', 'anxiety', 'worried', 'overwhelmed']):
            return """
 **Sentiment**: Stress/Anxiety
 **Confidence**: 85.0%

**Detailed Probabilities**:
- Stress/Anxiety: 85.0% ████████████████
- Neutral/Normal: 15.0% ███
"""
        else:
            return """
 **Sentiment**: Neutral/Normal
 **Confidence**: 80.0%

**Detailed Probabilities**:
- Stress/Anxiety: 20.0% ████
- Neutral/Normal: 80.0% ████████████████
"""
    
    try:
        # Tokenize input
        inputs = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=128,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get predicted class and confidence
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item() * 100
        
        # Format output
        output = f"""
 **Sentiment**: {LABEL_NAMES[predicted_class]}
 **Confidence**: {confidence:.1f}%

**Detailed Probabilities**:
"""
        
        for i, label in enumerate(LABEL_NAMES):
            prob = probs[0][i].item() * 100
            bar_length = int(prob / 5)
            bar = " " * bar_length
            output += f"\n• {label}: {prob:.1f}% {bar}"
        
        # Add recommendations based on prediction
        if predicted_class == 0:  # Stress/Anxiety
            output += """

 **Recommendations**:
- Consider reaching out to a mental health professional
- Practice stress-reduction techniques (deep breathing, meditation)
- Take regular breaks and prioritize self-care
- Connect with supportive friends or family

 **Disclaimer**: This is an AI tool for educational purposes only, not a substitute for professional mental health support.
"""
        else:  # Neutral
            output += """

 **Recommendations**:
- Maintain your positive mental health habits
- Continue with regular self-care practices
- Stay connected with your support system
- Monitor your emotional well-being regularly
"""
        
        return output
        
    except Exception as e:
        return f" Error during prediction: {str(e)}"


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        #  Mental Health Sentiment Analyzer
        
        ### Analyze text for stress, anxiety, and emotional well-being indicators using AI
        
        This tool uses a fine-tuned **DistilBERT** model trained on 52,186 mental health statements,
        achieving **97.06% accuracy** in detecting mental health sentiment.
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label=" Enter your text",
                placeholder="Type or paste your thoughts, feelings, or any text here...",
                lines=6,
                max_lines=10
            )
            
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")
        
        with gr.Column(scale=2):
            output_text = gr.Markdown(
                label=" Analysis Results",
                value="*Results will appear here after analysis*"
            )
    
    gr.Markdown("###  Try these examples:")
    
    gr.Examples(
        examples=[
            ["I'm feeling really overwhelmed with work and can't seem to catch a break. Everything is stressing me out."],
            ["Had a wonderful day today! Feeling grateful for everything and everyone in my life."],
            ["Just finished my project. It was okay, nothing special."],
            ["I can't stop worrying about everything. My anxiety is through the roof and I feel hopeless."],
            ["Feeling peaceful and relaxed after a good meditation session. Taking things one day at a time."],
        ],
        inputs=input_text,
        label="Click an example to try it"
    )
    
    gr.Markdown(
        """
        ---
        
        ###  About This Tool
        
        **Model Details:**
        - Architecture: DistilBERT (66M parameters)
        - Training Data: 52,186 mental health statements
        - Test Accuracy: 97.06%
        - Precision: 98% | Recall: 98%
        
        **Important Disclaimer:**
        This tool is for **educational and research purposes only**. It should NOT be used as:
        - A substitute for professional mental health diagnosis
        - The sole basis for treatment decisions
        - Emergency mental health support
      
        
        ---
       
        """
    )
    
    # Event handlers
    analyze_btn.click(
        fn=predict_sentiment,
        inputs=input_text,
        outputs=output_text
    )
    
    clear_btn.click(
        fn=lambda: ("", "*Results will appear here after analysis*"),
        outputs=[input_text, output_text]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,  # Set to True for public link
        server_name="0.0.0.0",  # For deployment
        server_port=7860
    )
