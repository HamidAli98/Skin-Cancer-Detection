import streamlit as st
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Skin Condition Image Classifier", page_icon="🔬", layout="centered")
st.title("🔬 Skin Condition Image Classifier")
st.warning("Educational prototype only—not a medical diagnosis. If a lesion is changing, bleeding, painful, or concerning, consult a qualified clinician.")

MODEL_PATH = "skin_disease_cnn_model.joblib"
IMG_SIZE = (128, 128)

@st.cache_resource(show_spinner="Loading model…")
def load_model_bundle(path):
    bundle = joblib.load(path)
    if not all(k in bundle for k in ("config", "weights", "label_map")):
        raise ValueError("Model package is missing config, weights, or label_map.")
    model = tf.keras.Sequential.from_config(bundle["config"])
    model.build((None, 128, 128, 3))
    model.set_weights(bundle["weights"])
    labels = {int(v): str(k) for k, v in bundle["label_map"].items()}
    return model, labels

try:
    model, labels = load_model_bundle(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH}. Put it beside this app file in your repository.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

uploaded = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    resized = image.resize(IMG_SIZE)
    x = np.asarray(resized, dtype=np.float32)[None, ...]  # model Rescaling layer divides by 255
    with st.spinner("Analyzing image…"):
        probabilities = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probabilities))
    st.subheader("Prediction (experimental)")
    st.metric(labels.get(idx, f"Class {idx}"), f"{float(probabilities[idx])*100:.1f}% confidence")
    rows = sorted([(labels.get(i, f"Class {i}"), float(p)) for i, p in enumerate(probabilities)], key=lambda t: t[1], reverse=True)
    st.write("Class probability scores")
    st.bar_chart({name: score for name, score in rows})
    st.caption("Confidence is the model's output score, not the probability that a diagnosis is correct.")
