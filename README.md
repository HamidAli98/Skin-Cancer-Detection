# Skin Lesion Streamlit App

## Files
- `skincancerdetector.py`: Streamlit application.
- `requirements.txt`: Python dependencies.

## Deploy
1. Put both files in the root of your GitHub repository.
2. Keep the Google Drive dataset folder shared as **Anyone with the link – Viewer**.
3. In Streamlit Community Cloud, select `skincancerdetector.py` as the main file.
4. Deploy.

## Dataset layout expected
The app searches recursively for these folders:
- `melanoma`
- `nevus`
- `seborrheic_keratosis`

They may be nested under `train/` and a parent folder.

## Model requirement
The app does not train a model on startup. It can load:
- `skin_lesion_cnn_model.keras`
- `skin_lesion_cnn_model.h5`
- `skin_lesion_cnn_weights.joblib` (only if it is a complete compatible classifier)

The classifier must return exactly three class scores in this order:
1. melanoma
2. nevus
3. seborrheic_keratosis

A weights-only file or a binary classifier is not compatible with this three-class interface. Do not treat synthetic/random labels as medically meaningful.

This app is an educational demo, not a medical diagnostic tool.
