from __future__ import annotations

import ast
import os
from typing import Any, Dict, Optional

import requests
import streamlit as st
from PIL import Image


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def get_api_url() -> str:
    default = os.getenv("STREAMLIT_API_URL", "http://localhost:8000/predict")
    return st.sidebar.text_input("FastAPI /predict URL", value=default).rstrip("/")


def call_predict(endpoint: str, file_bytes: bytes, filename: str) -> Dict[str, Any]:
    try:
        resp = requests.post(
            endpoint,
            files={"file": (filename, file_bytes, "image/jpeg")},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc


def format_value(key: str, raw_val: Any) -> Optional[str]:
    if raw_val is None:
        return None

    if isinstance(raw_val, list):
        cleaned = [str(v).strip() for v in raw_val if str(v).strip() not in {"", "-", "–"}]
        return ", ".join(cleaned) if cleaned else None

    if isinstance(raw_val, str) and raw_val.strip().startswith("[") and raw_val.strip().endswith("]"):
        try:
            parsed = ast.literal_eval(raw_val)
            if isinstance(parsed, list):
                return format_value(key, parsed)
        except Exception:
            pass

    val = str(raw_val).strip()
    if val in {"", "-", "–"}:
        return None

    if key == "packaging_en":
        val = val.replace("[", "").replace("]", "").replace("'", "")
        parts = [p.strip() for p in val.split(",") if p.strip()]
        return ", ".join(parts) if parts else None

    # Don't convert PLUS to +
    if key == "environmental_score_grade":
        return val

    return val or None



def show_product_info(title: str, data: Optional[Dict[str, Any]], *, image_key: str = "image_url"):
    st.subheader(title)

    if not data:
        if title.startswith("🌿"):
            st.info("No greener alternative found with a better eco-score. 🌱")
        else:
            st.info("No product details to display.")
        return


    img_col, info_col = st.columns([1, 2])

    with img_col:
        img_url = data.get(image_key)
        if img_url:
            st.image(img_url, caption="Product", use_container_width=True)

    with info_col:
        visible = False
        # Enforce display order here
        ordered_keys = [
            ("environmental_score_grade", "Eco Score"),
            ("packaging_en", "Packaging"),
            ("main_category", "Main Category"),
        ]
        for key, label in ordered_keys:
            pretty = format_value(key, data.get(key))
            if pretty:
                visible = True
                st.write(f"**{label}:** {pretty}")

    if not visible:
        st.write("No product details to display.")


# ──────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="GreenChoice Predictor", page_icon="🌿", layout="centered")

st.title("🌿 GreenChoice Product Advisor")
st.caption("Upload a product photo to see its eco-score, packaging, and a greener alternative.")

api_url = get_api_url()

uploaded = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        image = Image.open(uploaded)
        st.subheader("Uploaded Product")
        left, right = st.columns([1, 2])
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with right:
            st.success("Image ready. Click **Predict** to analyse.")
    except Exception:
        st.error("Unable to display the selected image.")

    if st.button("Predict", type="primary"):
        with st.spinner("Analysing…"):
            try:
                result = call_predict(api_url, uploaded.getvalue(), uploaded.name)
            except RuntimeError as err:
                st.error(str(err))
            else:
                show_product_info("📦 Product Details", result.get("prediction"))
                show_product_info("🌿 Greener Alternative", result.get("greener_alternative"))
else:
    st.info("👆 Drag & drop or browse to select an image.")
