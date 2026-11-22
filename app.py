import os
from pathlib import Path

import streamlit as st

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="DermaScan | Skin Image Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).resolve().parent / "skin_disease_cnn_model.joblib"
IMG_SIZE = (128, 128)

# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #14263d;
        --muted: #687b91;
        --teal: #087f8c;
        --teal-dark: #066875;
        --pale: #eff8f8;
        --line: #e3ebf1;
        --surface: #ffffff;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(8,127,140,.08), transparent 26rem),
            #f5f8fb;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(245,248,251,.88); }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -.035em;
    }
    .topbar {
        display:flex; align-items:center; justify-content:space-between;
        gap:16px; margin-bottom:1.5rem;
    }
    .brand {
        display:flex; align-items:center; gap:12px;
        font-family:'Manrope',sans-serif; font-weight:800;
        font-size:1.12rem; color:var(--ink);
    }
    .brand-mark {
        width:42px; height:42px; border-radius:14px;
        display:flex; align-items:center; justify-content:center;
        color:white; font-size:21px;
        background:linear-gradient(145deg,#0b8792,#14546f);
        box-shadow:0 8px 20px rgba(8,127,140,.20);
    }
    .top-pill {
        display:inline-flex; align-items:center; gap:7px;
        padding:7px 12px; border-radius:999px;
        background:#e8f5f3; color:#08736f;
        font-size:.78rem; font-weight:700;
        border:1px solid #d2ebe7;
    }
    .hero {
        background:linear-gradient(115deg,#102b46 0%,#124d62 58%,#087f8c 100%);
        border-radius:24px; padding:30px 34px;
        color:white; margin-bottom:22px;
        box-shadow:0 14px 36px rgba(19,57,79,.13);
        position:relative; overflow:hidden;
    }
    .hero:after {
        content:""; position:absolute; width:260px; height:260px;
        border:1px solid rgba(255,255,255,.13); border-radius:50%;
        right:-65px; top:-110px;
        box-shadow:0 0 0 28px rgba(255,255,255,.035),
                   0 0 0 58px rgba(255,255,255,.025);
    }
    .hero-kicker {
        text-transform:uppercase; letter-spacing:.16em;
        color:#a9e1df; font-size:.72rem; font-weight:700;
        margin-bottom:10px;
    }
    .hero h1 {
        color:white !important; font-size:clamp(1.8rem,3.5vw,2.55rem);
        line-height:1.15; margin:0 0 10px 0;
    }
    .hero p {
        color:#d4e8ed; max-width:660px;
        font-size:.98rem; line-height:1.65; margin:0;
    }
    .section-heading {
        font-family:'Manrope',sans-serif; font-size:1.18rem;
        font-weight:800; color:var(--ink); margin:8px 0 4px 0;
    }
    .section-sub {
        color:var(--muted); font-size:.88rem; margin-bottom:14px;
    }
    .panel {
        background:var(--surface); border:1px solid var(--line);
        border-radius:18px; padding:20px;
        box-shadow:0 5px 18px rgba(27,55,80,.035);
        height:100%;
    }
    .panel-label {
        color:#71849a; font-size:.72rem; font-weight:700;
        text-transform:uppercase; letter-spacing:.12em;
        margin-bottom:8px;
    }
    .metric-label {
        color:#71849a; font-size:.78rem; font-weight:600;
        margin-bottom:5px;
    }
    .metric-value {
        font-family:'Manrope',sans-serif; font-size:1.55rem;
        color:var(--ink); font-weight:800; line-height:1.2;
    }
    .muted { color:var(--muted); font-size:.86rem; line-height:1.55; }
    .result-chip {
        display:inline-block; padding:6px 10px; border-radius:8px;
        background:#e8f5f3; color:#08736f; font-size:.78rem;
        font-weight:700; margin-bottom:10px;
    }
    .notice {
        border:1px solid #f0dfb7; background:#fff9eb;
        border-radius:14px; padding:13px 16px;
        color:#76551b; font-size:.82rem; line-height:1.55;
        margin:18px 0;
    }
    .footer {
        border-top:1px solid var(--line); margin-top:34px;
        padding-top:16px; color:#8493a4; font-size:.76rem;
        display:flex; justify-content:space-between; gap:12px;
    }
    div[data-testid="stFileUploader"] {
        background:#f8fbfd; border:1.5px dashed #b9cbd8;
        border-radius:14px; padding:12px;
    }
    div[data-testid="stFileUploader"]:hover { border-color:#087f8c; }
    .stButton > button {
        border-radius:11px; min-height:46px; font-weight:700;
        border:0; color:white;
        background:linear-gradient(100deg,#087f8c,#12647a);
        box-shadow:0 7px 16px rgba(8,127,140,.17);
    }
    .stButton > button:hover {
        color:white; border:0;
        background:linear-gradient(100deg,#066875,#104f67);
    }
    [data-testid="stImage"] img { border-radius:13px; }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding:1rem 1rem 2rem; }
        .hero { padding:24px 22px; border-radius:18px; }
        .footer { flex-direction:column; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Header and introduction
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="brand-mark">✚</div>
        <span>DermaScan <span style="font-weight:500;color:#71849a">/ AI Lab</span></span>
      </div>
      <div class="top-pill">● &nbsp; MODEL INTERFACE</div>
    </div>
    <div class="hero">
      <div class="hero-kicker">Computer vision · Research prototype</div>
      <h1>Skin image analysis</h1>
      <p>Upload a skin image to explore the output of the trained image-classification
      model. The result is an experimental model prediction, not a medical assessment.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Model loading - lazy, cached
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_bundle(model_path: str):
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

    # The training notebook stores label_map as {label: index}.
    labels = {int(index): str(label) for label, index in bundle["label_map"].items()}
    return model, labels


# -------------------------------------------------------------------
# Upload and analyze
# -------------------------------------------------------------------
st.markdown('<div class="section-heading">01 &nbsp; Upload an image</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Choose a clear JPG, JPEG, or PNG image. '
    'For best results, use a well-lit, focused photograph.</div>',
    unsafe_allow_html=True,
)

upload_col, guide_col = st.columns([1.65, 1], gap="large")

with upload_col:
    uploaded = st.file_uploader(
        "Select skin image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Supported formats: JPG, JPEG and PNG.",
    )

with guide_col:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-label">Image guidelines</div>
          <div style="font-weight:700;color:#14263d;margin-bottom:8px">A useful image is:</div>
          <div class="muted">✓ In focus and well lit<br>
          ✓ Showing the area clearly<br>
          ✓ Not heavily filtered or edited<br>
          ✓ In JPG, JPEG, or PNG format</div>
          <div style="height:10px"></div>
          <div class="muted">Avoid including personal information in the image.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if uploaded is None:
    st.markdown(
        """
        <div class="notice">
          <b>Before you begin:</b> This educational prototype cannot diagnose skin cancer
          or any other condition. Do not use its output to make treatment decisions or
          delay medical care.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="panel" style="text-align:center;padding:30px 18px">
          <div style="font-size:2rem;margin-bottom:8px">⌁</div>
          <div style="font-weight:800;color:#14263d">Your image preview will appear here</div>
          <div class="muted">Upload an image above to begin the analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    from PIL import Image
    import numpy as np

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be opened as an image. Please try another JPG or PNG.")
        st.stop()

    st.markdown('<div class="section-heading">02 &nbsp; Review and analyze</div>', unsafe_allow_html=True)
    preview_col, result_col = st.columns([1.05, 1], gap="large")

    with preview_col:
        st.markdown('<div class="panel"><div class="panel-label">Image preview</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.caption(f"Image size: {image.width} × {image.height} pixels")
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-label">Analysis workspace</div>
              <div style="font-weight:800;font-size:1.1rem;color:#14263d;margin-bottom:7px">
                Ready to analyze
              </div>
              <div class="muted">The model will resize the image to 128 × 128 pixels
              and calculate its class scores.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        analyze = st.button("✦  Analyze image", use_container_width=True)

    if analyze:
        if not MODEL_PATH.is_file():
            st.error(
                f"Model file not found: {MODEL_PATH.name}. "
                "Place it in the same repository folder as app.py."
            )
            st.stop()

        try:
            with st.spinner("Loading the trained model…"):
                model, labels = load_model_bundle(str(MODEL_PATH))

            resized = image.resize(IMG_SIZE)
            x = np.asarray(resized, dtype=np.float32)[None, ...]

            with st.spinner("Analyzing image…"):
                probabilities = model.predict(x, verbose=0)[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_label = labels.get(predicted_index, f"Class {predicted_index}")
            confidence = float(probabilities[predicted_index])

            st.markdown('<div class="section-heading">03 &nbsp; Model output</div>', unsafe_allow_html=True)
            metric_col, note_col = st.columns([1, 1.6], gap="large")

            with metric_col:
                st.markdown(
                    f"""
                    <div class="panel">
                      <div class="panel-label">Top predicted class</div>
                      <div class="result-chip">EXPERIMENTAL OUTPUT</div>
                      <div class="metric-value">{predicted_label}</div>
                      <div style="height:12px"></div>
                      <div class="metric-label">Model score</div>
                      <div style="font-family:Manrope,sans-serif;font-size:2rem;font-weight:800;color:#087f8c">
                        {confidence * 100:.1f}%
                      </div>
                      <div class="muted">This is the model's output score, not a
                      probability that the person has a disease.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with note_col:
                st.markdown(
                    """
                    <div class="panel">
                      <div class="panel-label">How to interpret this result</div>
                      <div style="font-weight:700;color:#14263d;margin-bottom:7px">
                        An experimental classification only
                      </div>
                      <div class="muted">The model selects the class with the highest
                      output score among the categories it was trained on. It may be
                      wrong, especially for images or conditions not represented in
                      its training data.</div>
                      <div style="height:12px"></div>
                      <div class="muted"><b>Next step:</b> If you have a skin concern,
                      consult a qualified healthcare professional.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-heading">Class score breakdown</div>', unsafe_allow_html=True)
            rows = sorted(
                [
                    (labels.get(i, f"Class {i}"), float(score))
                    for i, score in enumerate(probabilities)
                ],
                key=lambda item: item[1],
                reverse=True,
            )

            score_left, score_right = st.columns(2, gap="large")
            for index, (name, score) in enumerate(rows):
                target = score_left if index % 2 == 0 else score_right
                with target:
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'font-size:.87rem;font-weight:700;color:#30465d;margin:9px 0 4px">'
                        f'<span>{name}</span><span>{score * 100:.1f}%</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(max(0.0, min(1.0, score)))

            st.markdown(
                """
                <div class="notice">
                  <b>Medical safety notice:</b> This application is an academic
                  machine-learning demonstration, not a medical device. Its output
                  must not be used for diagnosis, treatment, or reassurance. Seek
                  professional medical advice for any changing, bleeding, painful,
                  or concerning skin lesion.
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as exc:
            st.error(f"Could not load or run the model: {type(exc).__name__}: {exc}")
            with st.expander("Technical error details"):
                st.exception(exc)

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
      <span>DermaScan · Academic AI project</span>
      <span>Experimental image classification · Not for clinical use</span>
    </div>
    """,
    unsafe_allow_html=True,
)
