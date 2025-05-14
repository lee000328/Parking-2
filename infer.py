# infer.py

from ultralytics import YOLO
import cv2

# 모델 로딩
model = YOLO('c:/Users/82010/runs/detect/train10/weights/best.pt')  # ← 경로는 실제 폴더명으로 변경

# 이미지 경로
img_path = 'test_images/sample.jpg' # 이미지파일 이름, 형식따라 변경

# 추론
results = model(img_path)

# 결과 시각화
for result in results:
    result.save(filename='result.jpg')  # 시각화 이미지 저장
    result.show()                       # 창으로 보기
