import torch
import tempfile
import shutil
from app.utils import save_upload_file

model = torch.load("models/best.pt", map_location="cpu")
model.eval()

async def run_detection(file):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        await save_upload_file(file, temp.name)
        img = temp.name

    results = model(img)
    return results.pandas().xyxy[0].to_dict(orient="records")
