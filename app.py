import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import sys
import os
import subprocess

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")

# Debugging info
st.caption(f"Python executable: `{sys.executable}`")

# --- Load or Train Model
MODEL_PATH = "saved_model"

if not os.path.exists(MODEL_PATH):
    st.warning("No trained model found. Training model now... ⏳")
    
    # Run main.py to train the model
    try:
        subprocess.run(["python", "main.py"], check=True)
        st.success("✅ Model trained successfully.")
    except subprocess.CalledProcessError as e:
        st.error("❌ Failed to train the model. Check your `main.py`.")
        st.stop()

# Now load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
st.caption(f"TensorFlow Version: {tf.__version__}")

# --- Prediction Function
def predict_sentiment(text: str):
    prediction = model.predict(tf.constant([text]))[0][0]
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
    confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)
    return sentiment, confidence, prediction

def plot_confidence(pred):
    fig, ax = plt.subplots()
    bar_color = 'green' if pred > 0.5 else 'red'
    ax.bar(["Negative", "Positive"], [1 - pred, pred], color=[('grey' if pred > 0.5 else bar_color), (bar_color if pred > 0.5 else 'grey')])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    st.pyplot(fig)

# --- User input (Text box)
st.subheader("📥 Analyze a Single Review")
review_text = st.text_area("Enter your movie review:")

if st.button("Analyze"):
    if not review_text.strip():
        st.warning("Please enter a valid review.")
    else:
        sentiment, confidence, raw_pred = predict_sentiment(review_text)
        st.success(f"**Sentiment:** {sentiment} | **Confidence:** {confidence}%")
        plot_confidence(raw_pred)

# --- Upload multiple reviews from .txt
st.subheader("📄 Upload .txt File for Batch Analysis")
uploaded_file = st.file_uploader("Upload a .txt file with one review per line", type=["txt"])

if uploaded_file:
    content = uploaded_file.read().decode('utf-8')
    reviews = [line.strip() for line in content.split('\n') if line.strip()]

    if reviews:
        st.write(f"Found {len(reviews)} reviews.")
        results = []
        for i, review in enumerate(reviews):
            sentiment, confidence, _ = predict_sentiment(review)
            results.append((review, sentiment, confidence))
        
        st.subheader("📊 Results")
        for i, (review, sentiment, confidence) in enumerate(results):
            st.write(f"**Review {i+1}:** _{review}_")
            st.write(f"→ **Sentiment:** {sentiment} | **Confidence:** {confidence}%")
            st.markdown("---")
    else:
        st.warning("The file is empty or badly formatted.")
