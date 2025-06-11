from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.detect import run_detection
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "X-Ray YOLOv8 API is live 🚀"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    result = await run_detection(file)
    return JSONResponse(content=result)

# optional for local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
