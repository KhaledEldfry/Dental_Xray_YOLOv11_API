from fastapi import FastAPI, File, UploadFile
from app.predict import run_prediction
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = await run_prediction(file)
    return JSONResponse(content=result)
