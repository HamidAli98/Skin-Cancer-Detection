import os
import glob
import warnings
import zipfile
import shutil
from pathlib import Path

import gdown
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
)

DRIVE_ZIP_ID = "1VT6VZcRypoK65on706Qoqn7jCFec95HR"
ZIP_PATH = Path("skin_lesion_dataset.zip")
DATA_DIR = Path("dataset")
MODEL_CANDIDATES = [
    Path("skin_lesion_cnn_model.keras"),
    Path("skin_lesion_cnn_model.h5"),
    Path("skin_lesion_cnn_weights.joblib"),
]
CLASS_NAMES = ["melanoma", "nevus", "seborrheic_keratosis"]
IMAGE_SIZE = (128, 128)


# ---------------------------------------------------------
# Dataset download and discovery
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_dataset():
    """Download the shared dataset ZIP from Google Drive and extract it."""
    # Reuse an existing dataset if class folders already contain images.
    if find_class_directories():
        return str(DATA_DIR)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with st.spinner("Downloading dataset ZIP from Google Drive..."):
            downloaded = gdown.download(
                id=DRIVE_ZIP_ID,
                output=str(ZIP_PATH),
                quiet=False,
            )
        if not downloaded or not ZIP_PATH.exists() or ZIP_PATH.stat().st_size == 0:
            raise RuntimeError("Google Drive did not return a valid ZIP file.")

        # Validate ZIP before extracting, and avoid path traversal entries.
        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            bad_member = None
            root = DATA_DIR.resolve()
            for member in archive.namelist():
                target = (DATA_DIR / member).resolve()
                if target != root and root not in target.parents:
                    bad_member = member
                    break
            if bad_member:
                raise RuntimeError(f"Unsafe ZIP path detected: {bad_member}")
            archive.extractall(DATA_DIR)

    except Exception as exc:
        raise RuntimeError(
            "Could not download or extract the Google Drive ZIP. Confirm the file is "
            "shared as 'Anyone with the link – Viewer' and that the link is valid. "
            f"Google Drive/gdown details: {exc}"
        ) from exc


    # Some ZIPs contain an extra top-level directory; class discovery walks recursively.
    if not find_class_directories():
        raise RuntimeError(
            "ZIP downloaded and extracted, but no class image folders were found. "
            "Expected folders named melanoma, nevus, and seborrheic_keratosis."
        )

    return str(DATA_DIR)


def find_class_directories():
    """Find class directories, including when Drive adds a parent folder."""
    found = {}
    if not DATA_DIR.exists():
        return found

    for current, dirs, files in os.walk(DATA_DIR):
        folder_name = Path(current).name.lower().strip()
        if folder_name in CLASS_NAMES:
            image_files = [
                f for f in files
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
            if image_files:
                found[folder_name] = Path(current)

    return found


def count_images(class_dirs):
    counts = {}
    for class_name in CLASS_NAMES:
        folder = class_dirs.get(class_name)
        if folder:
            counts[class_name] = sum(
                1 for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
        else:
            counts[class_name] = 0
    return counts


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading classification model...")
def load_model():
    """Load a complete Keras model or a compatible joblib model."""
    for model_path in MODEL_CANDIDATES:
        if not model_path.exists():
            continue

        if model_path.suffix in {".keras", ".h5"}:
            try:
                from tensorflow.keras.models import load_model as keras_load_model
                return keras_load_model(model_path), model_path.name, "keras"
            except Exception as exc:
                warnings.warn(f"Could not load {model_path}: {exc}")

        if model_path.suffix == ".joblib":
            try:
                model = joblib.load(model_path)
                if hasattr(model, "predict") or hasattr(model, "predict_proba"):
                    return model, model_path.name, "joblib"
            except Exception as exc:
                warnings.warn(f"Could not load {model_path}: {exc}")

    return None, None, None


def prepare_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def predict_image(model, model_type, image):
    arr = prepare_image(image)

    if model_type == "keras":
        raw = np.asarray(model.predict(np.expand_dims(arr, axis=0), verbose=0))
        scores = raw.reshape(-1)
    else:
        # Generic sklearn-style classifiers usually expect flattened features.
        features = arr.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            scores = np.asarray(model.predict_proba(features)).reshape(-1)
        else:
            scores = np.asarray(model.predict(features)).reshape(-1)

    if len(scores) != len(CLASS_NAMES):
        raise ValueError(
            f"The loaded model returned {len(scores)} output value(s), but this "
            f"dataset has {len(CLASS_NAMES)} classes. A compatible three-class "
            "model is required. The old binary model cannot classify all three "
            "categories."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid prediction values.")

    # Convert logits or arbitrary scores into a normalized display distribution.
    if np.any(scores < 0) or not np.isclose(float(np.sum(scores)), 1.0, atol=0.05):
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        scores = exp_scores / np.sum(exp_scores)
    else:
        scores = scores / np.sum(scores)

    return scores


# ---------------------------------------------------------
# User interface
# ---------------------------------------------------------
st.title("🔬 Skin Lesion Classification")
st.caption("Three-class image classification demo")

st.warning(
    "This application is for educational research only. It is not a medical "
    "diagnostic tool. Do not use its predictions to make health decisions. "
    "Consult a qualified dermatologist for assessment."
)

with st.sidebar:
    st.header("Dataset")
    st.write("Google Drive dataset folder")
    st.code(DRIVE_FOLDER_ID)
    if st.button("Download / refresh dataset", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    with st.spinner("Checking Google Drive dataset..."):
        download_dataset()
    class_dirs = find_class_directories()
    counts = count_images(class_dirs)

    if not any(counts.values()):
        st.error(
            "No class images were found. The expected structure is "
            "`train/melanoma`, `train/nevus`, and "
            "`train/seborrheic_keratosis`. Check the shared folder link "
            "and its access permissions."
        )
    else:
        st.subheader("Dataset overview")
        cols = st.columns(3)
        for col, class_name in zip(cols, CLASS_NAMES):
            col.metric(class_name.replace("_", " ").title(), counts[class_name])

        missing = [name for name in CLASS_NAMES if counts[name] == 0]
        if missing:
            st.warning("Missing or empty class folder(s): " + ", ".join(missing))

except Exception as exc:
    st.error(str(exc))
    st.stop()

st.divider()
st.subheader("Upload a skin lesion image")
uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, PNG, or WEBP image",
    type=["jpg", "jpeg", "png", "webp"],
)

model, model_name, model_type = load_model()

if model is None:
    st.info(
        "The dataset is connected, but a compatible trained three-class model "
        "was not found. The repository currently contains "
        "`skin_lesion_cnn_weights.joblib`; this file must be verified before it "
        "can be used. A binary model or weights-only file cannot automatically "
        "classify melanoma, nevus, and seborrheic keratosis."
    )
    st.caption(
        "Add a complete three-class Keras model named "
        "`skin_lesion_cnn_model.keras` to the repository, or provide a compatible "
        "joblib classifier. Do not rename weights-only files to pretend they "
        "are complete models."
    )
else:
    st.success(f"Loaded model: {model_name}")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)

        if model is not None:
            with st.spinner("Analyzing image..."):
                scores = predict_image(model, model_type, image)
            best_idx = int(np.argmax(scores))

            with right:
                st.subheader("Model output")
                st.metric(
                    "Highest model score",
                    CLASS_NAMES[best_idx].replace("_", " ").title(),
                    f"{scores[best_idx] * 100:.1f}%",
                )
                st.write("Scores by class")
                for class_name, score in zip(CLASS_NAMES, scores):
                    st.write(f"**{class_name.replace('_', ' ').title()}** — {score * 100:.1f}%")
                    st.progress(float(score))
        else:
            with right:
                st.info(
                    "Image uploaded successfully. Prediction is unavailable "
                    "until a compatible three-class trained model is supplied."
                )
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

st.divider()
st.caption(
    "Important: Dataset folder names are treated as class labels. Verify the "
    "dataset annotations and model training procedure before interpreting any output."
)
