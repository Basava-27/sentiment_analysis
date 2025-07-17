import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import emoji
import re

@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("sentiment_model_3class")
    tokenizer = DistilBertTokenizerFast.from_pretrained("sentiment_model_3class")
    return model, tokenizer

def preprocess(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

model, tokenizer = load_model()
model.eval()

st.title("Amazon Review Sentiment Classifier")
review = st.text_area("Enter your review:")

if st.button("Classify"):
    review = preprocess(review)
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    st.success(f"Predicted Sentiment: **{label_map[pred]}**")
