import torch
import tempfile
import shutil
from app.utils import save_upload_file


from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})


from ultralytics import YOLO
model = YOLO("app/models/best.pt") 

model.eval()

async def run_detection(file):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        await save_upload_file(file, temp.name)
        img = temp.name

    results = model(img)
    return results[0].boxes.pandas().to_dict(orient="records")
