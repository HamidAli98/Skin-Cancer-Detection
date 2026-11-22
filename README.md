# Skin Lesion Classifier (Streamlit)

1. Run `train_skin_model_pytorch.ipynb` in Google Colab (T4 GPU) and download `skin_model.pt`.
2. Put `skin_model.pt` in this folder next to `app.py`.
3. Local: `pip install -r requirements.txt && streamlit run app.py`
4. Deploy: push `app.py`, `requirements.txt`, `skin_model.pt` to GitHub, then create the app at share.streamlit.io.

Educational use only. Not a medical device.
