
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI(title="NEO Issue Category Predictor")

with open("linear_svm_model.pkl", "rb") as f:
    model = pickle.load(f)

class IssueRequest(BaseModel):
    description: str

@app.get("/")
def home():
    return {"message": "NEO Issue Category API is running"}

@app.post("/predict")
def predict_issue(request: IssueRequest):
    text = request.description.strip()

    if not text:
        return {"error": "Please enter an issue description"}

    category = model.predict([text])[0]

    return {
        "issue_description": text,
        "predicted_category": category
    }
