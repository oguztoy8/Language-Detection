import streamlit as st
import joblib


st.title("Language Detection")

@st.cache_resource
def load_model():
    return joblib.load('language_pipeline.pkl')

pipeline = load_model()

text = st.text_area("Enter text to detect language:", height=100)

if st.button("Detect Language"):
    if text:
        prediction = pipeline.predict([text])[0]
        st.success(f"Detected Language: **{prediction}**")
    else:
        st.warning("Please enter some text")