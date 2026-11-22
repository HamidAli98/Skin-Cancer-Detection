import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Skin Condition Image Classifier",
    page_icon="🔬",
    layout="centered",
)

st.title("🔬 Skin Condition Image Classifier")
st.warning(
    "Educational prototype only—not a medical diagnosis. If a lesion is "
    "changing, bleeding, painful, or concerning, consult a qualified clinician."
)

MODEL_PATH = Path(__file__).resolve().parent / "skin_disease_cnn_model.joblib"
IMG_SIZE = (128, 128)


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_path: str):
    # Import heavy ML libraries only when a user actually requests a prediction.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import joblib
    import tensorflow as tf

    bundle = joblib.load(model_path)
    required = ("config", "weights", "label_map")
    if not isinstance(bundle, dict) or not all(k in bundle for k in required):
        raise ValueError("Model package is missing config, weights, or label_map.")

    model = tf.keras.Sequential.from_config(bundle["config"])
    model.build((None, IMG_SIZE[0], IMG_SIZE[1], 3))
    model.set_weights(bundle["weights"])
    labels = {int(index): str(name) for name, index in bundle["label_map"].items()}
    return model, labels


uploaded = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    from PIL import Image
    import numpy as np

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if not MODEL_PATH.is_file():
        st.error(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place it in the same repository folder as app.py."
        )
        st.stop()

    try:
        with st.spinner("Loading the model for the first time…"):
            model, labels = load_model_bundle(str(MODEL_PATH))

        resized = image.resize(IMG_SIZE)
        x = np.asarray(resized, dtype=np.float32)[None, ...]

        with st.spinner("Analyzing image…"):
            probabilities = model.predict(x, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))
        st.subheader("Prediction (experimental)")
        st.metric(
            labels.get(predicted_index, f"Class {predicted_index}"),
            f"{float(probabilities[predicted_index]) * 100:.1f}% confidence",
        )

        rows = sorted(
            [
                (labels.get(i, f"Class {i}"), float(score))
                for i, score in enumerate(probabilities)
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        st.write("Class probability scores")
        st.bar_chart({name: score for name, score in rows})
        st.caption(
            "Confidence is the model's output score, not the probability "
            "that a diagnosis is correct."
        )
    except Exception as exc:
        st.error(f"Could not load or run the model: {type(exc).__name__}: {exc}")
        st.exception(exc)
