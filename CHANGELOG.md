# 📘 업데이트 일지 (Changelog)

## 2025-05-02
- 프로젝트 초기 구조 세팅 완료
  - `.gitignore`, `README.md`, `requirements.txt` 작성
  - Git 저장소 초기화 및 GitHub 연결
## 2025-05-07
- 경로변경 및 가상환경 생성
  - Onedrive로 인한 권한 오류로 인해 폴더 변경
  - 폴더 변경이후 가상환경 제작
## 2025-05-08
- 라벨링 작업 및 업로드시도
  - roboflow를 이용한 라벨링 작업
  - 업로드 도중 문제현상 발견(해결완료)
  - train 폴더만 존재 vaild 폴더 존재x 및 디스크 공간부족
## 2025-05-09
- 첫 업로드 및 검증시도, 학습과정시 gpu사용 연결
  - 작업도중 건물형 지상주차장의 경우 지하와 비슷한 형태 인지(추후서술예정)
  - vaild 이미지 하나로 테스트 및 제대로 인식 완료
  - gpu 연결도중 중간중간 경로변경으로 인한 오류 발생
  - 경로 통합 및 교체x
## 2025-05-10
- 새 리포지트리 생성
  - 기존 리포지트리 커밋 과정 실수로 용량부족으로 커밋 실패
  - 기존은 .git 폴더 그대로 백업 및 새 리포지트리로 이동
## 2025-05-12
- 인식결과 분류 및 인식대상 바운딩박스 표기
  - main.py와 infer.py에 각 코드용도 설명
  - sample 이미지 테스트 인식실패
  - 13장이라는 너무 적은 학습용량
  - yolo에서 학습된 데이터 사용 혹은 직접 라벨링 후 학습시키기
## 2025-05-13
- roboflow universe 공용 데이터 다운 및 학습
  - main.py 와 infer.py 수정
  - 여전히 sample이미지 테스트 인식 실패
## 2025-05-14
- roboflow universe 공용데이터 추가 수집
  - conf값을 0.1로 낮춘경우 pillar 한개 인식
  - 계속해서 추가 데이터 학습을 통해 높은 인식도를 목표
## 2025-05-16
- 추가 데이터 학습 진행
  - 기둥인식 실패
  - 빛을 기둥으로 인식/ 학습데이터 검토 및 초기화 후 직접 촬영 혹은 양질의 데이터셋으로 재 학습 예정
  - 앙상블 추론을 통해 학습내용을 공유 문제 발생시 추후 분리예정
## 2025-05-19
- 기존데이터 노이즈, 밝기조절등을 이용하여 재학습
  - 기존학습과 추가 재학습 데이터를 사용
  - pillar의 인식을 성공, rebar의 학습부족으로 인식실패
  - 추후 rebar 혹은 자연광등 추가 학습을 통해 건물형과 지하분리예정
  - 기존 main과 infer의 통합버전인 run.py 생성
  - 아래 추가코드는 run.py를 터미널에 입력해서 각 제목대로 나옴
  - main.py와 infer.py또한 기존대로 실행가능

# 시각화 포함 추론
  python run.py --mode infer --weights weights/best1.pt weights/best2.pt weights/best3.pt weights/best4.pt weights/best5.pt weights/best6.pt weights/best7.pt --image test_images/sample.jpg

# 단순 분류 판단만 수행 (터미널 출력)
  python run.py --mode classify --weights weights/*.pt --image test_images/sample.jpg

# 시각화 + 저장 + 출력
  python run.py --mode infer --weights weights/*.pt --image test_images/sample.jpg
