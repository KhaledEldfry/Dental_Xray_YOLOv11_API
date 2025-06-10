from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
import io

app = FastAPI()

# Load model
model = YOLO("yolov11.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    annotated_frame = results[0].plot()

    # Convert to streamable image
    _, im_png = cv2.imencode(".png", annotated_frame)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
