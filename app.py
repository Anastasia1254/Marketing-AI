from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from collections import Counter
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

app = Flask(__name__)

# ----------------------------
# Load model
# ----------------------------
MODEL_PATH = "sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# ----------------------------
# Sentiment prediction
# ----------------------------
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()

    sentiment = "Positive" if torch.argmax(probs).item() == 1 else "Negative"

    return sentiment, {
        "negative": round(probs[0].item() * 100, 2),
        "positive": round(probs[1].item() * 100, 2)
    }

# ----------------------------
# Marketing analytics
# ----------------------------
def extract_keywords(text, top_n=6):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {'the','and','is','in','to','a','of','it','i','this','for','with','on','was','but'}
    words = [w for w in words if w not in stopwords and len(w) > 3]
    return Counter(words).most_common(top_n)

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def emotion_scores(text):
    t = text.lower()
    return {
        "joy": len(re.findall(r'great|love|excellent|amazing|perfect', t)),
        "anger": len(re.findall(r'bad|awful|terrible|hate|worst', t)),
        "trust": len(re.findall(r'reliable|safe|secure|quality', t))
    }

def buyer_persona(sentiment, length):
    if sentiment == "Negative" and length > 40:
        return "Critical Reviewer"
    if sentiment == "Positive":
        return "Brand Advocate"
    return "Neutral Observer"

def marketing_insight(sentiment, keywords):
    if sentiment == "Negative":
        return f"⚠️ High risk feedback. Improve: {', '.join(k[0] for k in keywords[:3])}"
    return "✅ Positive brand perception. Use as testimonial or ad copy."

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    data = {}

    if request.method == "POST":
        text = request.form.get("review_text")

        sentiment, scores = predict_sentiment(text)
        keywords = extract_keywords(text)
        emotions = emotion_scores(text)

        data = {
            "text": text,
            "sentiment": sentiment,
            "scores": scores,
            "keywords": keywords,
            "language": detect_language(text),
            "persona": buyer_persona(sentiment, len(text.split())),
            "insight": marketing_insight(sentiment, keywords),
            "emotions": emotions
        }

    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)
