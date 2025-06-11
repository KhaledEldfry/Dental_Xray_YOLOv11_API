import torch
import tempfile
import os
from app.utils import save_upload_file
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from fastapi import UploadFile

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

async def run_detection(file: UploadFile):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            await save_upload_file(file, temp.name)
            img_path = temp.name
            print(f"Image saved temporarily at: {img_path}")

        print("Running detection...")
        results = model(img_path)
        
        output_path = img_path.replace(".jpg", "_output.jpg")
        results[0].save(filename=output_path)
        print(f"Detection results saved at: {output_path}")

        try:
            os.unlink(img_path)
            print("Temporary file cleaned up")
        except Exception as e:
            print(f"Warning: Could not delete temp file {img_path}: {str(e)}")

        return output_path

    except Exception as e:
        print(f"Detection error: {str(e)}")
        
        if 'img_path' in locals() and os.path.exists(img_path):
            try:
                os.unlink(img_path)
            except:
                pass
                
        raise
