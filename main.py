from infer import detect_objects  # 감지 함수 불러오기

def classify_parking(detected_labels):
    """
    감지된 객체를 기반으로 주차장 유형을 판단합니다.
    :param detected_labels: YOLO가 감지한 클래스 이름들의 집합
    :return: 주차장 분류 결과 문자열
    """
    if 'rebar' in detected_labels and 'pillar' in detected_labels:
        return "건물형 또는 지하주차장"
    else:
        return "기타 또는 일반 주차장"

def main():
    image_path = 'sample.jpg'  # 분석할 이미지 경로 (루트에 있어야 함)
    labels = detect_objects(image_path)  # 객체 감지
    print("감지된 객체:", labels)  # 감지 결과 출력

    parking_type = classify_parking(labels)  # 주차장 유형 판별
    print("주차장 종류:", parking_type)

if __name__ == "__main__":
    main()  # 메인 함수 실행
