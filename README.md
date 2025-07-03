# 🌿 SafiPack Product Advisor


SafiPack Product Advisor is an AI-powered tool built using Streamlit and FastAPI to identify the **packaging content** of food and household products from images. Designed for use in Kigali, Rwanda, it supports environmentally conscious shopping by analyzing packaging components and suggesting alternatives with more considerate packaging. Users simply upload an image, and the app provides immediate insights based on visual packaging data.



## Why SafiPack?

Consumers in Kigali want clearer, quicker information when choosing environmentally considerate products. SafiPack uses image recognition and open datasets to support better packaging awareness—no barcode needed.


## Getting Started

Follow the steps below to clone and run the project locally:

```bash
# 1. Clone the repository
$ git clone https://github.com/your-org/SafiPack.git
$ cd SafiPack

# 2. Create and activate a virtual environment
$ python -m venv .venv
$ source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Start the FastAPI backend (on port 8000)
$ uvicorn app:app --reload

# 5. In a new terminal, start the Streamlit app (on port 8501)
$ streamlit run SafiPack_streamlit_app.py
```

Once both servers are running, open [http://localhost:8501](http://localhost:8501) in your browser. Upload a product image and click **Predict** to analyze.

---

## Project Structure

```
├─ app.py                       # FastAPI – /predict endpoint
├─ SafiPack_streamlit_app.py # Streamlit UI
├─ models/                      # Torch / ONNX model weights
├─ data/                        # Reference product database (CSV + images)
└─ requirements.txt
```

