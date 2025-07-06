from __future__ import annotations

import gzip
import io
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

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
# Environmental grade normalization
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
# Globals to be loaded on startup
# ---------------------------------------------------------------------------
_model: Optional[torch.jit.ScriptModule] = None
_label_encoders: Optional[Dict[str, Any]] = None
_multi_encoders: Optional[Dict[str, Any]] = None
_single_label_cols: List[str] = []
_multi_label_cols: List[str] = []
_catalog: Optional[pd.DataFrame] = None

def load_compressed_pickle(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------------------------------------------------------
# Image transform
# ---------------------------------------------------------------------------
_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def _decode_prediction(outs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    pred: Dict[str, Any] = {}
    assert _label_encoders is not None
    assert _multi_encoders is not None

    for col in _single_label_cols:
        idx = outs[col].argmax(1).item()
        pred[col] = _label_encoders[col].classes_[idx]

    for col in _multi_label_cols:
        scores = outs[col].sigmoid().squeeze(0).cpu().numpy()
        mlb = _multi_encoders[col]
        chosen = [cls for cls, s in zip(mlb.classes_, scores) if s > THRESHOLD]
        pred[col] = chosen

    return pred

def _predict_image(img_bytes: bytes) -> Dict[str, Any]:
    if _model is None:
        raise RuntimeError("Model not loaded")
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Invalid image format or corrupted file.")

    x = _TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outs = _model(x)
    return _decode_prediction(outs)

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
    if _catalog is None:
        return None

    candidates = _catalog[
        (_catalog["main_category_en"] == cat) &
        (_catalog["env_rank"] > cur_rank) &
        _catalog["labels_en"].apply(lambda x: _labels_match(x, cur_labels))
    ]

    if candidates.empty:
        return None

    best = candidates.sort_values(
        by=["env_rank", "nutriscore_grade", "quantity_in_grams"],
        ascending=[False, True, True]
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
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="GreenChoice Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global _model, _label_encoders, _multi_encoders, _single_label_cols, _multi_label_cols, _catalog
    print("Loading model and encoders at startup...")
    _model = torch.jit.load(MODEL_DIR / "model_quantized_scripted.pt", map_location=DEVICE)
    _model.eval()

    enc = load_compressed_pickle(MODEL_DIR / "encoders.pkl.gz")
    _label_encoders = enc["label_encoders"]
    _multi_encoders = enc["multi_encoders"]
    _single_label_cols = enc["SINGLE_LABEL_COLS"]
    _multi_label_cols = enc["MULTI_LABEL_COLS"]

    df = pd.read_csv(DATA_DIR / "df_filtered_unique_complete_qtygrams.csv.gz", compression="gzip")
    df["environmental_score_grade"] = df["environmental_score_grade"].apply(_norm_grade)
    df = df.dropna(subset=["environmental_score_grade", "main_category_en"])
    df["env_rank"] = df["environmental_score_grade"].map(_VALID_GRADES)
    _catalog = df
    print("Startup load complete.")

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/")
def get_message() -> Dict[str, str]:
    return {"message": "Welcome to the GreenChoice Predictor API! Use POST /predict to send your image for prediction."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    try:
        img_bytes = await file.read()
        pred = _predict_image(img_bytes)
        alt = _recommend_alternative(pred)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

    return JSONResponse({"prediction": pred, "greener_alternative": alt})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)
