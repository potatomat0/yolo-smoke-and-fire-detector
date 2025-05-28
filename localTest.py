from huggingface_hub import hf_hub_download
import joblib
from ultralytics import YOLO 

REPO_ID = "kappH/NhanDienKhoiLua" 
FILENAME = "best.pt"  

model = YOLO(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))

# test ảnh bất kỳ 

# model.predict(source='example5.jpeg', imgsz=640, conf=0.25, save=True)

# test webcam

model.predict(source=0, imgsz=640, conf=0.25, show=True)

