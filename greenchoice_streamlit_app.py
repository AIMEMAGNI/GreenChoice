"""
GreenChoice Streamlit Client
===========================
A lightweight Streamlit interface that consumes the GreenChoice FastAPI
`/predict` endpoint. Upload a JPG/PNG image of a product and instantly get:

* **Predicted labels** for every single‑label & multi‑label task.
* **Greener alternative** suggestion (if available) with key details & image.

Usage
-----
1. Install dependencies (in your Streamlit environment):

   ```bash
   pip install streamlit requests pillow
   ```

2. Make sure the FastAPI service is running (locally or remote).
3. Run the app:

   ```bash
   STREAMLIT_API_URL=https://your-domain/predict streamlit run greenchoice_streamlit_app.py
   ```

   – or set the URL in the sidebar UI after launching.
"""
from __future__ import annotations

import io
import os
from typing import Any, Dict, Optional

import requests
import streamlit as st
from PIL import Image

###############################################################################
# Configuration & helpers
###############################################################################

def get_api_url() -> str:
    """Resolve the `/predict` endpoint URL from env or sidebar input."""
    default = os.getenv("STREAMLIT_API_URL", "http://localhost:8000/predict")
    return st.sidebar.text_input("FastAPI /predict URL", value=default).rstrip("/")


def call_predict(endpoint: str, file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Send image to the FastAPI `/predict` endpoint and return JSON response."""
    try:
        response = requests.post(
            endpoint,
            files={"file": (filename, file_bytes, "image/jpeg")},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc


def show_prediction(pred: Dict[str, Any]):
    """Pretty‑print the model predictions in two columns."""
    single_cols = {}
    multi_cols = {}
    for k, v in pred.items():
        if isinstance(v, list):
            multi_cols[k] = v
        else:
            single_cols[k] = v

    st.subheader("Single‑label predictions")
    if single_cols:
        for k, v in single_cols.items():
            st.write(f"**{k.replace('_', ' ').title()}**: {v}")
    else:
        st.info("No single‑label predictions returned.")

    st.subheader("Multi‑label predictions")
    if multi_cols:
        for k, labels in multi_cols.items():
            if labels:
                st.write(f"**{k.replace('_', ' ').title()}**: {', '.join(labels)}")
            else:
                st.write(f"**{k.replace('_', ' ').title()}**: –")
    else:
        st.info("No multi‑label predictions returned.")


def show_alternative(alt: Optional[Dict[str, Any]]):
    """Display greener alternative information if available."""
    st.subheader("Greener alternative")
    if alt is None:
        st.info("No greener product found with a better environmental score and overlapping labels.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        if alt.get("image_url"):
            st.image(alt["image_url"], caption="Suggested product", use_column_width=True)
    with col2:
        for field in [
            "brands_en",
            "environmental_score_grade",
            "nutriscore_grade",
            "packaging_en",
            "labels_en",
        ]:
            val = alt.get(field)
            if val is not None and val != "":
                pretty = field.replace("_en", "").replace("_", " ").title()
                st.write(f"**{pretty}:** {val}")

###############################################################################
# Streamlit UI
###############################################################################

st.set_page_config(page_title="GreenChoice Predictor", page_icon="🌿", layout="centered")

st.title("🌿 GreenChoice Image Prediction & Recommendation")
st.caption("Upload a product photo to get eco insights and greener alternatives.")

api_url = get_api_url()

uploaded = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_column_width=True)
    except Exception:
        st.error("Unable to display the selected image.")

    if st.button("Predict", type="primary"):
        with st.spinner("Contacting model & fetching predictions…"):
            try:
                file_bytes = uploaded.getvalue()
                result = call_predict(api_url, file_bytes, uploaded.name)
            except RuntimeError as e:
                st.error(str(e))
            else:
                show_prediction(result.get("prediction", {}))
                show_alternative(result.get("greener_alternative"))
else:
    st.info("👆 Drag & drop or browse to select an image.")
