# 🌿 GreenChoice Product Advisor

GreenChoice Product Advisor is a lightweight AI-powered tool built using Streamlit and FastAPI to identify the **packaging content** of food and household products from images. Designed for use in Kigali, Rwanda, it supports environmentally conscious shopping by analyzing packaging components and suggesting local alternatives with more considerate packaging. Users simply upload an image, and the app provides immediate insights based on visual packaging data.


Watch a quick demo ▶️ **[Demo Video](https://youtu.be/mHXSji4aah8)**

---

## Why GreenChoice?

Consumers in Kigali want clearer, quicker information when choosing environmentally considerate products. GreenChoice uses image recognition and open datasets to support better packaging awareness—no barcode needed.

---

## Getting Started

Follow the steps below to clone and run the project locally:

```bash
# 1. Clone the repository
$ git clone https://github.com/your-org/greenchoice.git
$ cd greenchoice

# 2. Create and activate a virtual environment
$ python -m venv .venv
$ source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Start the FastAPI backend (on port 8000)
$ uvicorn app:app --reload

# 5. In a new terminal, start the Streamlit app (on port 8501)
$ streamlit run greenchoice_streamlit_app.py
```

Once both servers are running, open [http://localhost:8501](http://localhost:8501) in your browser. Upload a product image and click **Predict** to analyze.

---

## Project Structure

```
├─ app.py                       # FastAPI – /predict endpoint
├─ greenchoice_streamlit_app.py # Streamlit UI
├─ models/                      # Torch / ONNX model weights
├─ data/                        # Reference product database (CSV + images)
└─ requirements.txt
```

