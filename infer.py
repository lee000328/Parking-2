from ultralytics import YOLO
import cv2
import os

# 사전 학습된 YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # yolov8n.pt: 속도 빠른 경량 모델

def detect_objects(image_path):
    """
    이미지에서 객체를 감지하고 시각화 이미지도 저장합니다.
    :param image_path: 분석할 이미지 경로
    :return: 감지된 클래스 이름 집합
    """
    results = model(image_path)  # YOLO 모델로 객체 감지 수행
    result = results[0]  # 첫 번째 결과 가져오기

    detected_labels = set()
    image = cv2.imread(image_path)

    for box in result.boxes:
        cls_id = int(box.cls)  # 감지된 클래스 번호
        label = model.names[cls_id]  # 클래스 이름으로 변환
        detected_labels.add(label)

        # 시각화: 바운딩 박스 그리기
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
        conf = float(box.conf[0])  # 신뢰도

        # 바운딩 박스와 라벨 출력
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(image, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 시각화된 이미지 저장
    save_path = os.path.splitext(image_path)[0] + "_pred.jpg"
    cv2.imwrite(save_path, image)
    print(f"[+] 시각화 이미지 저장 완료: {save_path}")

    return detected_labels
