from pathlib import Path
import streamlit as st
import joblib
from train_model import train_and_save_model, MODEL_PATH, METRICS_PATH

st.set_page_config(page_title="SMS Spam Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ SMS Spam Detection")
st.write("Enter a text message below. The machine learning model will predict whether it is spam or legitimate (ham).")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        train_and_save_model()
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error("The model could not be prepared. Check your internet connection and try again.")
    st.exception(exc)
    st.stop()

message = st.text_area("Enter your SMS message", height=150, placeholder="Example: Congratulations! You have won a free prize...")
if st.button("Check Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        class_index = list(model.classes_).index(prediction)
        confidence = float(probabilities[class_index]) * 100

        if prediction == "spam":
            st.error("🚨 Prediction: SPAM")
        else:
            st.success("✅ Prediction: NOT SPAM (HAM)")
        st.metric("Model confidence", f"{confidence:.1f}%")
        st.caption("Confidence is the model's estimated probability, not a guarantee. Always use caution with suspicious links and requests.")

with st.expander("How does this project work?"):
    st.markdown("""
    1. **Text preprocessing:** the message is prepared for analysis.
    2. **TF-IDF:** words are converted into numerical features. Words that are useful for distinguishing messages receive higher weights.
    3. **Logistic Regression:** the classifier uses those features to predict `spam` or `ham`.
    4. **Prediction:** the app displays the predicted class and estimated confidence.
    """)

st.divider()
st.caption("Academic project: TF-IDF + Logistic Regression | Dataset: UCI SMS Spam Collection")
