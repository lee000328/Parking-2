import argparse
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision.ops import nms
from glob import glob


from glob import glob

def load_models(weight_glob_pattern):
    if isinstance(weight_glob_pattern, list):
        weight_glob_pattern = weight_glob_pattern[0]  # 리스트면 첫 번째 값 사용
    weight_paths = sorted(glob(weight_glob_pattern))
    if not weight_paths:
        raise FileNotFoundError(f"No .pt files found matching: {weight_glob_pattern}")
    return [YOLO(w) for w in weight_paths]


def run_inference(models, image_path):
    """모든 모델로 이미지 추론하고 결과 병합"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    all_boxes, all_scores, all_classes = [], [], []

    for model in models:
        results = model.predict(source=img_rgb, conf=0.25, iou=0.45, verbose=False)
        result = results[0]
        all_boxes.append(result.boxes.xyxy.cpu().numpy())
        all_scores.append(result.boxes.conf.cpu().numpy())
        all_classes.append(result.boxes.cls.cpu().numpy())

    boxes = np.vstack(all_boxes)
    scores = np.hstack(all_scores)
    classes = np.hstack(all_classes)

    keep = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.5)

    return img, boxes[keep.numpy()], scores[keep.numpy()], classes[keep.numpy()], models[0].names

def visualize(img, boxes, scores, classes, names, save_path='ensemble_result.jpg'):
    """결과 이미지 시각화 및 저장"""
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(save_path, img)
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def classify(classes, names):
    """주차장 유형 분류"""
    detected = [names[int(cls)] for cls in classes]
    print(f"감지된 클래스: {detected}")
    if any(cls in detected for cls in ['pillar', 'rebar', 'wall']):
        print("🅿️ 지하 혹은 건물형 주차장입니다.")
    else:
        print("🅿️ 일반 주차장입니다.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['infer', 'classify'], default='classify',
                        help="작업 모드 선택: 'infer'(이미지 시각화) or 'classify'(종류 분류)")
    parser.add_argument('--weights', nargs='+', required=True, help="YOLO 가중치 경로들")
    parser.add_argument('--image', type=str, required=True, help="입력 이미지 경로")
    args = parser.parse_args()

    models = load_models(args.weights)
    img, boxes, scores, classes, names = run_inference(models, args.image)

    if args.mode == 'infer':
        visualize(img, boxes, scores, classes, names)
    elif args.mode == 'classify':
        classify(classes, names)

if __name__ == "__main__":
    main()
