from model import load_model
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    review: str

model = load_model()

@app.post("/predict")
def predict(data: InputData):
    review = data.review

    response = model(review)

    return {"Prediction": response[0]["label"], "Confidence": f"{round(response[0]['score'] * 100, 2)}%"}