from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.detect import run_detection
from fastapi.responses import FileResponse
import uvicorn
import os

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "X-Ray YOLOv11 API is live 🚀"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_path = await run_detection(file)
    return FileResponse(image_path, media_type="image/jpeg", filename="result.jpg")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "port": os.getenv("PORT", "10000")
    }
# optional for local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
