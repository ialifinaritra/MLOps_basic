from fastapi import FastAPI
from inference import ColaPredictor
import uvicorn

app = FastAPI(title="Tuto MLOPs")
predictor = ColaPredictor("./models/best-checkpoint.ckpt")


@app.get("/")
async def home():
    return "<h2>This is a sample NLP Project</h2>"


@app.post("/predict/{sentence}")
async def predict(sentence: str):
    result = predictor.predict(sentence)

    return result


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)