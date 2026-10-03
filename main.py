from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
from fastapi.responses import HTMLResponse

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
app = FastAPI(title="Iris Flower Prediction API")

# Load model
try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    pipeline = None

# Request schema with intervals
class IrisRequest(BaseModel):
    sepal_length: float = Field(..., ge=4.3, le=7.9)
    sepal_width: float = Field(..., ge=2.0, le=4.4)
    petal_length: float = Field(..., ge=1.0, le=6.9)
    petal_width: float = Field(..., ge=0.1, le=2.5)

class IrisResponse(BaseModel):
    prediction: str
    confidence: float | None = None

@app.get("/")
def root():
    return {"message": "Iris prediction API is running. See /docs for usage."}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.post("/predict", response_model=IrisResponse)
def predict(request: IrisRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    features = [[request.sepal_length, request.sepal_width,
                 request.petal_length, request.petal_width]]
    pred = pipeline.predict(features)[0]
    confidence = float(pipeline.predict_proba(features).max()) if hasattr(pipeline, "predict_proba") else None
    return IrisResponse(prediction=pred, confidence=confidence)

@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Iris Flower Tester</title>
        <style>
            body { font-family: sans-serif; max-width: 500px; margin: 60px auto; }
            input { width: 100%; padding: 8px; margin: 5px 0; }
            button { padding: 10px; margin-top: 10px; width: 100%; }
            img { margin-top: 15px; width: 300px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <h2>Iris Flower Predictor</h2>
        <form id="irisForm">
            <label>Sepal Length (4.3–7.9):</label>
            <input type="number" step="0.1" id="sepal_length" min="4.3" max="7.9" required>
            <label>Sepal Width (2.0–4.4):</label>
            <input type="number" step="0.1" id="sepal_width" min="2.0" max="4.4" required>
            <label>Petal Length (1.0–6.9):</label>
            <input type="number" step="0.1" id="petal_length" min="1.0" max="6.9" required>
            <label>Petal Width (0.1–2.5):</label>
            <input type="number" step="0.1" id="petal_width" min="0.1" max="2.5" required>
            <button type="submit">Predict</button>
        </form>
        <h3 id="result" style="margin-top:20px;"></h3>
        <div id="image"></div>
        <script>
        const speciesImages = {
            "setosa": "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_2.jpg",
            "versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
            "virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
        };

        document.getElementById("irisForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const sample = {
                sepal_length: parseFloat(document.getElementById("sepal_length").value),
                sepal_width: parseFloat(document.getElementById("sepal_width").value),
                petal_length: parseFloat(document.getElementById("petal_length").value),
                petal_width: parseFloat(document.getElementById("petal_width").value)
            };
            const res = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(sample)
            });
            const data = await res.json();
            document.getElementById("result").innerText =
                "Prediction: " + data.prediction + " (confidence: " + (data.confidence ? (data.confidence*100).toFixed(2) + "%" : "n/a") + ")";
            document.getElementById("image").innerHTML =
                "<img src='" + speciesImages[data.prediction.toLowerCase()] + "' alt='" + data.prediction + "'>";
        });
        </script>
    </body>
    </html>
    """
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
app = FastAPI(title="The Sentiment Analysis API")

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    pipeline = None

class SentimentaRequest(BaseModel):
    text: str = Field(..., min_length = 1, description = "Review text to analyze")

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float | None = None

@app.get("/")
def root():
    return {"message": "Sentiment analysis API is running. See /docs for usage."}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentaRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    pred = pipeline.predict([request.text])[0]
    label = "positive" if pred == 1 else "negitive"

    confidence = None
    if hasattr(pipeline, "predict_proba"):
        confidence = float(pipeline.predict_proba([request.text]).max())
    return SentimentResponse(sentiment = label, confidence = confidence)

from fastapi.responses import HTMLResponse

@app.get("/test", response_class = HTMLResponse)
def test_page():
    return""" """ 
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Tester</title>
    </head>
    <body style="font-family: sans-serif; max-width: 500px; margin: 60px auto;">
        <h2>Sentiment Analyzer</h2>
        <textarea id="text" rows="4" style="width:100%; font-size:16px;" placeholder="Type a sentence..."></textarea>
        <br>
        <br>
        <button onclick="predict()" style="padding:8px 16px; font-size:16px;">Predict</button>
        <h3 id="result" style="margin-top:20px;"></h3>
        <script>
        async function predict() {
            const text = document.getElementById("text").value;
            const res = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ text: text })
            });
            const data = await res.json();
            document.getElementById("result").innerText =
                data.sentiment + " (confidence: " + (data.confidence ?? "n/a") + ")";
        }
        </script>
    </body>
    </html>
    """
"""Simple FastAPI app serving a scikit-learn model.

Run locally:
    uvicorn main:app --reload

Test it:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
"""

"""from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(
    title="Getting Started with ML in Production API",
    description="A minimal prediction API built for the workshop.",
    version="1.0.0",
)

# Load the model once at startup
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Iris features: [sepal_length, sepal_width, petal_length, petal_width]"
    )


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str


IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


@app.get("/")
def root():
    return {"message": "Workshop ML API is running. See /docs for usage."}


@app.get("/health")
def health():
    #Basic health check endpoint — useful for deployment platforms & load balancers.
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Did you run the training notebook?")

    features = np.array(request.features).reshape(1, -1)
    pred = int(model.predict(features)[0])

    return PredictionResponse(prediction=pred, class_name=IRIS_CLASSES[pred])"""