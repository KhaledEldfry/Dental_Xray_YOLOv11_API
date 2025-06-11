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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        await save_upload_file(file, temp.name)
        img_path = temp.name

    results = model(img_path)

    # حفظ الصورة اللي فيها البوكسات
    output_path = img_path.replace(".jpg", "_output.jpg")
    results[0].save(filename=output_path)

    return output_path
