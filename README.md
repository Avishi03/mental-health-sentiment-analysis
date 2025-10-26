#  Mental Health Sentiment Analysis using NLP

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/-Transformers-yellow.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-97.06%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A deep learning NLP project that analyzes text for mental health sentiment classification. This system uses a fine-tuned **DistilBERT** model to detect stress, anxiety, and emotional well-being indicators from text, achieving **97.06% accuracy**.
**Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/Avishi0309/Sentimental_Analysis)

##  Project Overview

This project addresses mental health awareness by leveraging AI to analyze textual data. The model classifies text into two categories:
- **Stress/Anxiety/Negative** (0) - Mental health concerns including anxiety, stress, depression, suicidal thoughts
- **Neutral/Normal** (1) - Balanced emotional state

**Dataset**: 52,186 mental health statements from social media and surveys

##  Key Features

-  **Fine-tuned BERT Model** - DistilBERT optimized for mental health text classification
-  **97.06% Accuracy** - High-performance model validated on 10,438 test samples
-  **Interactive Web App** - Real-time sentiment analysis with Gradio interface
-  **Detailed Metrics** - Comprehensive classification reports with precision/recall
-  **Confidence Scoring** - Probabilistic predictions for each sentiment class
-  **Deployed on Hugging Face** - Accessible online demo for testing

##  Tech Stack

- **Language**: Python 3.8+
- **Framework**: PyTorch 2.0
- **Model**: DistilBERT (distilbert-base-uncased)
- **Libraries**: 
  - `transformers` 4.57.1 - Hugging Face Transformers
  - `datasets` - Data handling
  - `scikit-learn` - Metrics and evaluation
  - `pandas` & `numpy` - Data processing
  - `gradio` - Web interface
- **Platform**: Google Colab with T4 GPU
- **Deployment**: Hugging Face Spaces

##  Project Structure
```
mental-health-sentiment-analysis/
├── notebooks/
│   └── Mental_Health_Sentiment_Analysis.ipynb
├── models/
│   └── mental_health_model/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer files
├── app/
│   └── app.py                      # Gradio application
├── results/                        # Training checkpoints
├── requirements.txt
├── README.md
└── .gitignore
```

##  Getting Started

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/mental-health-sentiment-analysis.git
cd mental-health-sentiment-analysis

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Running the Gradio App
```bash
python app/app.py
```

#### Making Predictions
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load model
tokenizer = DistilBertTokenizer.from_pretrained('./mental_health_model')
model = DistilBertForSequenceClassification.from_pretrained('./mental_health_model')

# Predict
text = "I'm feeling overwhelmed and stressed about everything."
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
print(f"Stress/Anxiety: {probs[0][0]:.2%}")
print(f"Neutral: {probs[0][1]:.2%}")
```

##  Dataset Information

**Source**: [Kaggle - Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

- **Total Samples**: 52,186 (after preprocessing)
- **Training Set**: 41,748 samples (80%)
- **Test Set**: 10,438 samples (20%)
- **Original Classes**: Anxiety, Stress, Depression, Suicidal, Bipolar, Personality Disorder, Normal
- **Mapped Classes**: 
  - Class 0 (Negative): 36,317 samples (69.6%)
  - Class 1 (Neutral): 15,869 samples (30.4%)

##  Model Performance

| Metric | Stress/Anxiety | Neutral | Overall |
|--------|----------------|---------|---------|
| **Precision** | 98% | 96% | 97% |
| **Recall** | 98% | 94% | 97% |
| **F1-Score** | 98% | 95% | 97% |
| **Support** | 7,264 | 3,174 | 10,438 |

### Overall Metrics
- **Test Accuracy**: **97.06%**
- **Weighted Avg F1**: 0.97
- **Training Time**: ~15 minutes (Google Colab T4 GPU)
- **Total Epochs**: 3
- **Final Training Loss**: 0.086

### Sample Predictions

| Input Text | Predicted Sentiment | Confidence |
|------------|---------------------|------------|
| "I'm feeling really overwhelmed with work..." | Stress/Anxiety | 99.8% |
| "Had a wonderful day today! Feeling grateful..." | Neutral | 99.0% |
| "I can't stop worrying about everything..." | Stress/Anxiety | 99.9% |
| "Feeling peaceful and relaxed..." | Neutral | 99.7% |

##  Training Configuration
```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,  # Mixed precision training
    max_length=128
)
```

**Hardware**: Google Colab T4 GPU  
**Training Duration**: 14 minutes 21 seconds  
**Total Steps**: 7,830

##  Training Progress

- **Epoch 1**: Loss decreased from 0.58 → 0.14
- **Epoch 2**: Loss stabilized around 0.06-0.12
- **Epoch 3**: Final loss reached 0.02-0.04

##  Live Demo

Try the model live on Hugging Face Spaces:  
🔗 [https://huggingface.co/spaces/Avishi0309/Sentimental_Analysis](https://huggingface.co/spaces/Avishi0309/Sentimental_Analysis)

##  How It Works

1. **Input**: User enters text (tweet, post, or statement)
2. **Tokenization**: DistilBERT tokenizer converts text to tokens
3. **Model Inference**: Fine-tuned model predicts sentiment
4. **Output**: Returns classification with confidence scores

##  Model Architecture
```
DistilBERT (66M parameters)
├── Embedding Layer (768 dimensions)
├── 6 Transformer Encoder Layers
│   ├── Multi-Head Self-Attention
│   └── Feed-Forward Networks
├── Pre-Classifier Layer
└── Classification Head (2 classes)
```

##  Future Enhancements

-  Multi-class classification (separate anxiety, depression, stress)
-  Support for multiple languages
-  Real-time social media monitoring integration
-  Mobile app development
-  Explainability features (attention visualization)
-  Integration with mental health resources API

## Important Disclaimer

**This tool is for educational and research purposes only.** It should NOT be used as:
- A substitute for professional mental health diagnosis
- The sole basis for treatment decisions
- Emergency mental health support

**If you or someone you know is in crisis:**
- 🇺🇸 National Suicide Prevention Lifeline: **988**
- 🇺🇸 Crisis Text Line: Text **HOME** to **741741**
- 🇮🇳 AASRA: **91-22-27546669**
- International: [Find Help](https://www.iasp.info/resources/Crisis_Centres/)

##  License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

##  Author

**Avishi Agrawal**

##  Acknowledgments

- [Hugging Face](https://huggingface.co/) for Transformers library and Spaces hosting
- [Kaggle](https://www.kaggle.com/) for the mental health dataset
- [Google Colab](https://colab.research.google.com/) for free GPU resources
- Suchintika Sarkar for the mental health sentiment dataset

