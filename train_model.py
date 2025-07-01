
import pandas as pd
import joblib
import re
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


df = pd.read_csv("dataset.csv")

def clean_text(text):
    if not isinstance(text, str):
        return ""


    text = html.unescape(text)


    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  
    text = re.sub(r"\S+@\S+", "", text)                  
    text = re.sub(r"@\w+|#\w+", "", text)               

   
    text = re.sub(r"[\r\n\t\u200b\u200c\u200d]", " ", text)  
    text = re.sub(r"\s+", " ", text).strip() 

    return text


df["Text"] = df["Text"].apply(clean_text) 
df = df.dropna()


pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer='char', ngram_range=(1, 5))),
    ('model', RandomForestClassifier())
])

print("Training model...")
pipeline.fit(df['Text'], df['language'])


joblib.dump(pipeline, "language_pipeline.pkl")
print("Model saved as language_pipeline.pkl")
