import streamlit as st
import joblib
import re
import string

# Load model
lgbm  = joblib.load('lgbm_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le    = joblib.load('label_encoder.pkl')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# UI
st.set_page_config(page_title="Email Classifier", page_icon="📧")
st.title("📧 Email Classifier")
st.caption("Multi-class · 8 categories · LightGBM + Ensemble · 95%+ accuracy")

email = st.text_area("Paste your email here", height=150)

if st.button("Classify"):
    if email.strip():
        cleaned   = preprocess(email)
        vectorized = tfidf.transform([cleaned])
        pred      = lgbm.predict(vectorized)[0]
        proba     = lgbm.predict_proba(vectorized)[0]
        label     = le.inverse_transform([pred])[0]
        confidence = round(proba.max() * 100, 2)

        st.success(f"**Category: {label}**")
        st.metric("Confidence", f"{confidence}%")

        st.subheader("All category scores")
        for i, cls in enumerate(le.classes_):
            st.progress(float(proba[i]), text=f"{cls} — {round(proba[i]*100, 1)}%")
    else:
        st.warning("Please enter an email!")