from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io

app = FastAPI()

# Load the trained YOLO model
model = YOLO("best.pt")  # Make sure best.pt is in the same directory

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Run prediction
    results = model.predict(image, conf=0.25)

    # Draw results (this returns a BGR numpy array)
    result_img = results[0].plot()

    # Convert BGR to RGB
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Convert to BytesIO for HTTP response
    pil_img = Image.fromarray(result_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
    import uvicorn

    if __name__ == "__main__":
        uvicorn.run("main:app", host="0.0.0.0", port=8000)
