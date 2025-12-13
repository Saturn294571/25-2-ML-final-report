# 전산통계 기말 과제 : 머신러닝 프로젝트

### 데이터셋 : Insurance Premium Prediction

- Insurance Premium Prediction : https://www.kaggle.com/competitions/playground-series-s4e12/data
  - 나이, 성별, 흡연 여부, 체중(BMI), 키 등의 정보를 바탕으로 보험료를 예측하는 데이터셋
  - XGboost를 사용하여 모델을 학습하고, 테스트 데이터셋을 사용하여 보험료를 예측.

### 프로젝트 구조
```bash
ML_final_report/
├── data/               # 원본 및 가공된 데이터 (train.csv 등은 여기로 이동)
├── notebooks/          # EDA 및 실험용 Jupyter Notebook (.ipynb)
├── src/                # 재사용 가능한 소스 코드
│   ├── features/       # 데이터 전처리 및 피처 엔지니어링 스크립트
│   ├── models/         # 모델 학습 및 추론 스크립트
│   └── utils/          # 공통 유틸리티 함수
├── models/             # 학습된 모델 파일 (.pkl, .json 등) 저장
├── .env/               # API 키 등 민감한 정보 (Git 제외됨)
├── .gitignore          # Git 관리 제외 목록
└── README.md           # 프로젝트 설명
```# 25-2-ML-final-report
