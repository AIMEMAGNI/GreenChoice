"""
GreenChoice Image Prediction & Recommendation API
================================================

This FastAPI application exposes a **/predict** endpoint that accepts a JPG/PNG image
and returns JSON containing:

* **prediction** – values for every task your model predicts (single‑label and
  multi‑label columns).
* **greener_alternative** – the best matching product from the catalog with a
  strictly better environmental score grade and at least one overlapping label
  (or `null` if none is found).
"""

from __future__ import annotations

import io
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
IMG_SIZE = 224
THRESHOLD = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Utility: environmental score normalisation & ranking
# ---------------------------------------------------------------------------
_VALID_GRADES = {
    "a-plus": 6,
    "a": 5,
    "b": 4,
    "c": 3,
    "d": 2,
    "e": 1,
    "f": 0,
}

def _norm_grade(grade: Any) -> Optional[str]:
    if grade is None:
        return None
    g = str(grade).strip().lower().replace("+", "plus").replace(" ", "-")
    return g if g in _VALID_GRADES else None

# ---------------------------------------------------------------------------
# Model definition (identical to notebook)
# ---------------------------------------------------------------------------
class _Head(nn.Module):
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.fc(x)


class _MultiTask(nn.Module):
    def __init__(self, n_single: Dict[str, int], n_multi: Dict[str, int]):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        in_f = backbone.classifier[1].in_features  # type: ignore[index]
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.heads = nn.ModuleDict(
            {k: _Head(in_f, n) for k, n in {**n_single, **n_multi}.items()}
        )

    def forward(self, x):  # type: ignore[override]
        feats = self.backbone(x)
        return {k: h(feats) for k, h in self.heads.items()}


# ---------------------------------------------------------------------------
# Single initialisation block – load *once* at start‑up
# ---------------------------------------------------------------------------
with open(MODEL_DIR / "encoders.pkl", "rb") as f:
    enc = pickle.load(f)

_LABEL_ENCODERS = enc["label_encoders"]
_MULTI_ENCODERS = enc["multi_encoders"]
_SINGLE_LABEL_COLS: List[str] = enc["SINGLE_LABEL_COLS"]
_MULTI_LABEL_COLS: List[str] = enc["MULTI_LABEL_COLS"]

_single_sizes = {c: len(_LABEL_ENCODERS[c].classes_) for c in _SINGLE_LABEL_COLS}
_multi_sizes = {c: len(_MULTI_ENCODERS[c].classes_) for c in _MULTI_LABEL_COLS}

_MODEL = _MultiTask(_single_sizes, _multi_sizes).to(DEVICE)
_MODEL.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=DEVICE))
_MODEL.eval()

_catalog = pd.read_csv(DATA_DIR / "df_filtered_unique_complete_qtygrams.csv")
_catalog["environmental_score_grade"] = _catalog["environmental_score_grade"].apply(_norm_grade)
_catalog = _catalog.dropna(subset=["environmental_score_grade", "main_category_en"])
_catalog["env_rank"] = _catalog["environmental_score_grade"].map(_VALID_GRADES)

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

def _decode_prediction(outs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    pred: Dict[str, Any] = {}

    for col in _SINGLE_LABEL_COLS:
        idx = outs[col].argmax(1).item()
        pred[col] = _LABEL_ENCODERS[col].classes_[idx]

    for col in _MULTI_LABEL_COLS:
        scores = outs[col].sigmoid().squeeze(0).cpu().numpy()
        mlb = _MULTI_ENCODERS[col]
        chosen = [cls for cls, s in zip(mlb.classes_, scores) if s > THRESHOLD]
        pred[col] = chosen

    return pred


def _predict_image(img_bytes: bytes) -> Dict[str, Any]:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = _TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outs = _MODEL(x)
    return _decode_prediction(outs)

# ---------------------------------------------------------------------------
# Greener‑alternative logic
# ---------------------------------------------------------------------------

def _labels_match(row_labels: str | float, cur_labels: set[str]) -> bool:
    if pd.isna(row_labels) or not str(row_labels).strip():
        return False
    row_set = {l.strip() for l in str(row_labels).split(",")}
    return bool(cur_labels.intersection(row_set))


def _recommend_alternative(pred: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cat = pred.get("main_category_en")
    cur_grade = _norm_grade(pred.get("environmental_score_grade"))
    cur_labels = set(pred.get("labels_en", []))

    if not cat or not cur_grade:
        return None

    cur_rank = _VALID_GRADES.get(cur_grade)

    candidates = _catalog[
        (_catalog["main_category_en"] == cat)
        & (_catalog["env_rank"] > cur_rank)
        & _catalog["labels_en"].apply(lambda x: _labels_match(x, cur_labels))
    ]

    if candidates.empty:
        return None

    best = candidates.sort_values(
        by=["env_rank", "nutriscore_grade", "quantity_in_grams"],
        ascending=[False, True, True],
    ).iloc[0]

    return {
        "brands_en": best.get("brands_en"),
        "environmental_score_grade": best.get("environmental_score_grade"),
        "nutriscore_grade": best.get("nutriscore_grade"),
        "packaging_en": best.get("packaging_en"),
        "labels_en": best.get("labels_en"),
        "image_url": best.get("image_url"),
    }


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="GreenChoice Predictor", version="1.0.0")

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    img_bytes = await file.read()
    try:
        pred = _predict_image(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

    alt = _recommend_alternative(pred)
    return JSONResponse({"prediction": pred, "greener_alternative": alt})


# ---------------------------------------------------------------------------
# Run server (Render-compatible)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
