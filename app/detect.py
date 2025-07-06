import torch
import tempfile
import os
from app.utils import save_upload_file
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from fastapi import UploadFile
from PIL import Image  # لازم تستورد PIL عشان تحفظ الصورة

torch.serialization.add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})

print("Current working directory:", os.getcwd())
print("Model exists:", os.path.exists("app/models/best.pt"))

try:
    model = YOLO("app/models/best.pt")
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    raise

async def run_segmentation(file: UploadFile):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            await save_upload_file(file, temp.name)
            img_path = temp.name
            print(f"Image saved temporarily at: {img_path}")

        print("Running segmentation...")
        results = model(img_path)

        # ارسم الماسكات على الصورة
        img_with_masks = results[0].plot()

        output_path = img_path.replace(".jpg", "_output.jpg")
        im = Image.fromarray(img_with_masks)
        im.save(output_path)
        print(f"Segmentation results saved at: {output_path}")

        try:
            os.unlink(img_path)
            print("Temporary file cleaned up")
        except Exception as e:
            print(f"Warning: Could not delete temp file {img_path}: {str(e)}")

        return output_path

    except Exception as e:
        print(f"Segmentation error: {str(e)}")

        if 'img_path' in locals() and os.path.exists(img_path):
            try:
                os.unlink(img_path)
            except:
                pass

        raise
