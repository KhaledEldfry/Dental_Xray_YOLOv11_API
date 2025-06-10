import torch
from PIL import Image
import numpy as np
from io import BytesIO

def get_yolov11_model():
    return torch.hub.load("WongKinYiu/yolov11", "custom", path="app/models/best.pt", source="local")

async def load_image(file):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(image)
