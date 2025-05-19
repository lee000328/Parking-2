from ultralytics import YOLO
import cv2
import torch
import numpy as np
from torchvision.ops import nms
from pathlib import Path

# ✅ 1. 여러 개의 YOLO 모델 가중치 불러오기
weight_paths = [
    'weights/best1.pt',
    'weights/best2.pt',
    'weights/best3.pt',
    'weights/best4.pt',
    'weights/best5.pt',
    'weights/best6.pt',
    'weights/best7.pt'
]
models = [YOLO(w) for w in weight_paths]  # 각각의 모델 인스턴스 생성

# ✅ 2. 추론할 이미지 불러오기
image_path = 'test_images/sample.jpeg'  # 테스트할 이미지 경로
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # YOLO는 RGB 형식 사용

# ✅ 3. 모든 모델로 예측 수행하고 결과 수집
all_boxes, all_scores, all_classes = [], [], []

for model in models:
    result = model.predict(source=img_rgb, conf=0.25, iou=0.45, verbose=False)[0]
    
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    
    all_boxes.append(boxes)
    all_scores.append(scores)
    all_classes.append(classes)

# ✅ 4. 결과 병합 후 NMS로 중복 제거
all_boxes = np.vstack(all_boxes)
all_scores = np.hstack(all_scores)
all_classes = np.hstack(all_classes)

boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

final_boxes = all_boxes[keep.numpy()]
final_scores = all_scores[keep.numpy()]
final_classes = all_classes[keep.numpy()]

# ✅ 5. 클래스 이름 불러오기 (첫 번째 모델 기준으로)
names = models[0].names

# ✅ 6. 감지된 클래스 이름 확인
detected_classes = [names[int(cls)] for cls in final_classes]
print(f"감지된 클래스: {detected_classes}")

# ✅ 7. 지하/건물형 주차장 여부 판단
if any(cls in detected_classes for cls in ['pillar', 'rebar', 'wall']):
    print("🅿️ 지하 혹은 건물형 주차장입니다.")
else:
    print("🅿️ 일반 주차장입니다.")
