# -*- coding: utf-8 -*-
"""
Streamlit demo for the supplied ISICCNN model.

IMPORTANT: The supplied model was trained on brightness-derived pseudo-labels,
not dermatologist-confirmed cancer labels. It is therefore a software/demo
classifier only and must not be used to diagnose skin cancer.
"""
from pathlib import Path
import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "selected_cnn_model.joblib"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ISICCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Put selected_cnn_model.joblib "
            "in the same folder as cancer_detection.py."
        )
    # This is the user's own uploaded model file. Only load trusted model files.
    payload = joblib.load(MODEL_PATH)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("The model file does not contain the expected state_dict.")
    model = ISICCNN().to(DEVICE)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

st.set_page_config(page_title="Skin Image CNN Demo", page_icon="🔬", layout="centered")
st.title("Skin Image CNN Demo")
st.warning(
    "Research/demo only — this model was trained using image-brightness "
    "pseudo-labels, not verified benign/malignant diagnoses. It cannot detect "
    "or rule out cancer. Do not use its output for medical decisions."
)
st.write("Upload an image to run the supplied CNN model.")

uploaded = st.file_uploader("Choose a skin image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        with st.spinner("Running model..."):
            model = load_model()
            tensor = preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        predicted = int(np.argmax(probabilities))
        labels = ["Low brightness (class 0)", "High brightness (class 1)"]
        st.subheader("Model output")
        st.write(f"Predicted class: **{labels[predicted]}**")
        st.write(f"Class 0 score: **{probabilities[0]:.1%}**")
        st.write(f"Class 1 score: **{probabilities[1]:.1%}**")
        st.caption("These are scores for the brightness-derived classes, not cancer probabilities.")
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
else:
    st.info("Upload a JPG or PNG image to begin.")
