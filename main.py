# main.py 예시
from ultralytics import YOLO

model = YOLO('c:/Users/82010/runs/detect/train8/weights/best.pt')
results = model("test_images/sample.jpg") # 이미지파일 이름, 형식따라 변경

names = model.names  # 클래스 이름 (예: {0: 'pillar', 1: 'rebar'})
for result in results:
    detected_classes = [names[int(cls)] for cls in result.boxes.cls]

    if 'pillar' in detected_classes or 'rebar' in detected_classes:
        print("지하 혹은 건물형 주차장입니다.")
    else:
        print("일반 주차장입니다.")
