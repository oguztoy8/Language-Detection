
import pandas as pd
import joblib
import re
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("dataset.csv")

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # HTML entity temizle ve Unicode normalize et
    text = html.unescape(text)

    # URL, e-posta ve mention/hashtag gibi gürültülü ama dil dışı yapıları temizle
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URL
    text = re.sub(r"\S+@\S+", "", text)                  # E-posta
    text = re.sub(r"@\w+|#\w+", "", text)                # @mention ve #hashtag

    # Unicode kontrol karakterlerini ve fazla boşlukları kaldır
    text = re.sub(r"[\r\n\t\u200b\u200c\u200d]", " ", text)  # görünmeyen karakterleri temizle
    text = re.sub(r"\s+", " ", text).strip()  # fazla boşlukları azalt

    return text

# Clean text
df["Text"] = df["Text"].apply(clean_text) 
df = df.dropna()

# Create and train pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer='char', ngram_range=(1, 5))),
    ('model', RandomForestClassifier())
])

print("Training model...")
pipeline.fit(df['Text'], df['language'])

# Save model
joblib.dump(pipeline, "language_pipeline.pkl")
print("Model saved as language_pipeline.pkl")