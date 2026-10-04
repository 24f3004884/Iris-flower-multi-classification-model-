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
    return {"message": "Iris Flower Prediction API is running. See /docs for usage."}

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
            "setosa": "https://en.wikipedia.org/wiki/Iris_flower_data_set#/media/File:Kosaciec_szczecinkowaty_Iris_setosa.jpg",
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
