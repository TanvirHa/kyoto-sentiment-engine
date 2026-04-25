from fastapi import FastAPI
from fastapi.responses import FileResponse # ADD THIS
import joblib
from utils import tokenize_jp
import uvicorn

model = joblib.load('kyoto_model.pkl')
app = FastAPI()

@app.get("/")
def home():
    # This sends the HTML file to your browser
    return FileResponse("index.html")

@app.get("/predict")
def predict(review: str):
    prediction = model.predict([review])[0]
    return {"review": review, "sentiment": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)