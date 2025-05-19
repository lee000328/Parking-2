from ultralytics import YOLO
import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision.ops import nms  # Non-Maximum Suppression

# ✅ 1. 사용할 YOLO 모델 가중치 파일 경로 (weights 폴더 기준)
weight_paths = [
    'weights/best1.pt',
    'weights/best2.pt',
    'weights/best3.pt',
    'weights/best4.pt',
    'weights/best5.pt',
    'weights/best6.pt',
    'weights/best7.pt'
]  # 앙상블에 사용할 YOLO 모델 가중치들

# ✅ 2. 각 모델 로딩
models = [YOLO(w) for w in weight_paths]  # 각 모델 인스턴스 생성

# ✅ 3. 추론할 이미지 로드 및 RGB 변환
image_path = 'test_images/sample.jpg'     # 추론할 이미지 경로
img = cv2.imread(image_path)              # BGR 형식으로 이미지 로드
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # YOLO 모델은 RGB 입력 필요

# ✅ 4. 모든 모델로부터 추론 결과 수집
all_boxes = []    # 박스 좌표 모음
all_scores = []   # confidence 점수 모음
all_classes = []  # 클래스 ID 모음

for model in models:
    results = model.predict(source=img_rgb, conf=0.25, iou=0.45, save=False, verbose=False)
    result = results[0]  # 단일 이미지 예측 결과

    # 박스, confidence, 클래스 ID 추출
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    all_boxes.append(boxes)
    all_scores.append(scores)
    all_classes.append(classes)

# ✅ 5. 모든 모델 결과 병합
all_boxes = np.vstack(all_boxes)      # (N, 4) 전체 바운딩 박스 좌표
all_scores = np.hstack(all_scores)    # (N,) confidence 점수
all_classes = np.hstack(all_classes)  # (N,) 클래스 번호

# ✅ 6. 중복 박스를 제거하기 위한 NMS 수행
boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
scores_tensor = torch.tensor(all_scores, dtype=torch.float32)

# IoU 0.5 이상 중복 박스를 제거
keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

# NMS 통과한 최종 결과만 남김
final_boxes = all_boxes[keep_indices.numpy()]
final_scores = all_scores[keep_indices.numpy()]
final_classes = all_classes[keep_indices.numpy()]

# ✅ 7. 최종 예측 결과를 이미지에 시각화
for box, score, cls in zip(final_boxes, final_scores, final_classes):
    x1, y1, x2, y2 = map(int, box)  # 좌표 정수형 변환
    label = f"{int(cls)} {score:.2f}"  # 라벨: 클래스 번호 + 점수
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 박스 그리기
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 2)  # 라벨 텍스트 추가

# ✅ 8. 최종 이미지 저장 및 출력
cv2.imwrite('ensemble_result.jpg', img)  # 결과 이미지 저장
cv2.imshow('Ensemble Result', img)       # 화면에 결과 출력
cv2.waitKey(0)
cv2.destroyAllWindows()
