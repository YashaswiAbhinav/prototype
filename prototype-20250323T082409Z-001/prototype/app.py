from flask import Flask, request, render_template, jsonify
import joblib
import re
import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS

# For periodic scheduling and fetching news feeds
from apscheduler.schedulers.background import BackgroundScheduler
import feedparser
import requests  # If you need to get news from a REST API

app = Flask(__name__)
CORS(app)  # Enable CORS if your frontend needs it

# -------------------------------------------------
# 0. Define the Custom LSTMClassifier (for joblib)
# -------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return out

# Make sure joblib.load can find this custom class
globals()["LSTMClassifier"] = LSTMClassifier

# -------------------------------------------------
# 1. Helper: Preprocess the input text
# -------------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)       # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ', text).strip()   # Remove extra spaces
    return text

# -------------------------------------------------
# 2. Load the Combined Models
# -------------------------------------------------
# Replace these paths with the correct ones on your system.
combined_model_path1 = "D:/[Programs]/FakeNewDetection/combined_model.pkl"
combined_model_path2 = "D:/[Programs]/FakeNewDetection/combined_model_true.pkl"

try:
    combined_model1 = joblib.load(combined_model_path1)
    combined_model2 = joblib.load(combined_model_path2)
    print("Combined models loaded successfully.")
except Exception as e:
    print("Error loading models:", e)
    exit()

# -------------------------------------------------
# 3. Function to extract individual components
# -------------------------------------------------
def get_model_components(model_dict):
    return (
        model_dict["tfidf"],
        model_dict["svd"],
        model_dict["lr_model"],
        model_dict["rf_model"],
        model_dict["tokenizer_lstm"],
        model_dict["lstm_model"],
        model_dict["max_len"],
        model_dict.get("distilbert_model", None),
        model_dict.get("tokenizer_distilbert", None)
    )

components1 = get_model_components(combined_model1)
components2 = get_model_components(combined_model2)

# Use CPU for inference (change to "cuda" if you plan to use GPU and models support it)
device = torch.device("cpu")

# -------------------------------------------------
# 4. Define Prediction Functions for Each Component
# -------------------------------------------------
def predict_lr_prob(tfidf, lr_model, text):
    vec = tfidf.transform([preprocess_text(text)])
    return lr_model.predict_proba(vec)[0][1]

def predict_rf_prob(tfidf, svd, rf_model, text):
    vec = tfidf.transform([preprocess_text(text)])
    vec_reduced = svd.transform(vec)
    return rf_model.predict_proba(vec_reduced)[0][1]

def predict_lstm_prob(tokenizer_lstm, lstm_model, max_len, text):
    seq = tokenizer_lstm.texts_to_sequences([preprocess_text(text)])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    tensor_input = torch.tensor(padded, dtype=torch.long).to(device)
    lstm_model.eval()
    with torch.no_grad():
        output = lstm_model(tensor_input)
        return torch.sigmoid(output).item()

def predict_distilbert_prob(tokenizer_distilbert, distilbert_model, text):
    if tokenizer_distilbert and distilbert_model:
        inputs = tokenizer_distilbert(
            preprocess_text(text),
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=128
        )
        distilbert_model.eval()
        with torch.no_grad():
            logits = distilbert_model(**inputs).logits
            return torch.softmax(logits, dim=1).cpu().numpy()[0][1]
    return None

def predict_combined(model_components, text):
    tfidf, svd, lr_model, rf_model, tokenizer_lstm, lstm_model, max_len, distilbert_model, tokenizer_distilbert = model_components

    p_lr = predict_lr_prob(tfidf, lr_model, text)
    p_rf = predict_rf_prob(tfidf, svd, rf_model, text)
    p_lstm = predict_lstm_prob(tokenizer_lstm, lstm_model, max_len, text)
    p_distilbert = predict_distilbert_prob(tokenizer_distilbert, distilbert_model, text)

    # Average probabilities; if DistilBERT isn't available then use the other three.
    if p_distilbert is not None:
        avg_prob = (p_lr + p_rf + p_lstm + p_distilbert) / 4
    else:
        avg_prob = (p_lr + p_rf + p_lstm) / 3

    return p_lr, p_rf, p_lstm, p_distilbert, avg_prob

# -------------------------------------------------
# 5. Real-Time News Fetching and Classification
# -------------------------------------------------
# Create a global list to store fetched and classified news articles
fetched_news = []

def fetch_and_classify_news():
    global fetched_news
    # Reset the news list (or merge new articles as needed)
    fetched_news = []

    # Define one or more RSS feed URLs. You can add additional feeds here.
    rss_feeds = [
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "http://feeds.bbci.co.uk/news/rss.xml"
    ]
    
    for rss_url in rss_feeds:
        feed = feedparser.parse(rss_url)
        # Loop over each entry in the RSS feed
        for entry in feed.entries:
            title = entry.get('title', '')
            description = entry.get('description', '')
            published_at = entry.get('published', '')  # RSS feeds usually have a published field
            
            # Combine title and description for classification
            combined_text = title + " " + description
            # Get predictions from both combined models – you could average them
            p_lr1, p_rf1, p_lstm1, p_distilbert1, avg_prob1 = predict_combined(components1, combined_text)
            p_lr2, p_rf2, p_lstm2, p_distilbert2, avg_prob2 = predict_combined(components2, combined_text)
            meta_prob = (avg_prob1 + avg_prob2) / 2.0
            final_class = "FAKE NEWS" if meta_prob >= 0.5 else "REAL NEWS"
            
            # Prepare the news item dictionary.
            news_item = {
                "title": title,
                "description": description,
                "source": entry.get('link', 'Unknown'),
                "published_at": published_at,
                "classification": final_class,
                "probability": meta_prob
            }
            
            fetched_news.append(news_item)
    
    print(f"Fetched and classified {len(fetched_news)} news items.")

# Schedule the fetch_and_classify_news function to run periodically (e.g., every 15 minutes)
scheduler = BackgroundScheduler()
scheduler.add_job(func=fetch_and_classify_news, trigger="interval", minutes=15)
scheduler.start()

# Optionally, run it once at startup so that there is data immediately
fetch_and_classify_news()

# -------------------------------------------------
# 6. Define the Main Page Route (GET and POST)
# -------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    # This route handles a manual news article submission for classification.
    if request.method == 'POST':
        news_text = request.form.get("news_text", "").strip()
        if not news_text:
            return render_template("index.html", results=None, news_text="")
        # Classify the manually submitted text
        p_lr1, p_rf1, p_lstm1, p_distilbert1, avg_prob1 = predict_combined(components1, news_text)
        p_lr2, p_rf2, p_lstm2, p_distilbert2, avg_prob2 = predict_combined(components2, news_text)
        meta_prob = (avg_prob1 + avg_prob2) / 2.0
        final_class = "FAKE NEWS" if meta_prob >= 0.5 else "REAL NEWS"
        results = {
            "Meta_Ensemble": {
                "Final_Prediction": final_class,
                "Meta_Probability": meta_prob
            },
            "Model 1": {
                "LR": p_lr1,
                "RF": p_rf1,
                "LSTM": p_lstm1,
                "DistilBERT": p_distilbert1 if p_distilbert1 is not None else None
            },
            "Model 2": {
                "LR": p_lr2,
                "RF": p_rf2,
                "LSTM": p_lstm2,
                "DistilBERT": p_distilbert2 if p_distilbert2 is not None else None
            }
        }
        return render_template("index.html", results=results, news_text=news_text)
    
    return render_template("index.html", results=None, news_text="")

# -------------------------------------------------
# 7. Define a /news Endpoint for the JavaScript Fetch
# -------------------------------------------------
@app.route('/news', methods=['GET'])
def news():
    # Return the globally stored news articles (which are updated periodically)
    return jsonify(fetched_news)

# -------------------------------------------------
# 8. Run the Flask Application
# -------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
