from huggingface_hub import hf_hub_download
from ultralytics import YOLO 
import os 

REPO_ID = "kappH/NhanDienKhoiLua" 
FILENAME = "best.pt"  
CURRENT_PATH = os.getcwd()
PROJECT = './runs/detect'
NAME = 'predict'
model = YOLO(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))

# test ảnh bất kỳ 

model.predict(source='example5.jpeg', imgsz=640, conf=0.25, save=True, project=PROJECT, name=NAME)

# test webcam

# model.predict(source=0, imgsz=640, conf=0.25, show=True)

