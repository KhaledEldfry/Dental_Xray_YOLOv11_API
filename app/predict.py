import torch
from app.utils import load_image, get_yolov11_model

model = get_yolov11_model()

async def run_prediction(file):
    image = await load_image(file)
    results = model(image)
    return results.pandas().xyxy[0].to_dict(orient="records")
