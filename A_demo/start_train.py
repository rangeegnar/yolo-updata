import time
import warnings
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# 加载模型
model = YOLO("boosting/yolov8_cbam.yaml").load("yolov8n.pt")

# 训练模型
results = model.train(
    data='data.yaml',
    epochs=10,
    imgsz=640,
    device='cpu',
    workers=0,
    batch=4,
    cache=True
)

# 等待10秒
time.sleep(10)