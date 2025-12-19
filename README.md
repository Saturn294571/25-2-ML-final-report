# 전산통계 기말 과제 : 머신러닝 프로젝트

### 데이터셋 : Regression with an insurance Dataset (Kaggle)

- **Dataset**: Regression with an insurance Dataset (https://www.kaggle.com/competitions/playground-series-s4e12/data)
  - 나이, 성별, 흡연 여부, BMI 등의 정보를 바탕으로 보험료(Premium Amount)를 예측하는 데이터셋.
  - 다양한 머신러닝 모델(Linear Regression, XGBoost, MLP)을 사용하여 실험을 진행.

### 프로젝트 개요 및 요약

본 프로젝트는 보험료 예측을 위한 회귀 모델을 구축하고 성능을 최적화하는 과정을 담고 있습니다.

1.  **데이터 탐색 (EDA) & 전처리**:
    -   데이터의 극심한 양의 왜도(Right-Skewness) 확인 및 `np.log1p` 로그 변환 적용.
    -   결측치 처리 (평균/최빈값 대치) 및 범주형 변수 인코딩 (Ordinal, One-Hot).
    -   파생 변수 생성 (날짜 데이터 활용).

2.  **모델링 및 실험**:
    -   **Linear Regression (Baseline)**: MAE 650.78, $R^2$ -0.18. 비선형성을 파악하지 못해 성능 저조.
    -   **XGBoost (Tree-based)**: MAE 623.99, $R^2$ -0.15. 로그 변환 및 튜닝을 통해 소폭 개선되었으나, 고액 보험료 구간에서 예측 한계(Systematic Under-prediction) 발견.
    -   **MLP (Neural Network)**: MAE 633.52, $R^2$ -0.16. 딥러닝 모델 역시 비슷한 한계를 보여줌.

3.  **결론 및 한계**:
    -   모든 모델이 MAE 620~650 수준에서 정체됨.
    -   잔차 분석 결과, 데이터 자체의 정보 부족(핵심 변수 누락)이 성능 개선의 주요 걸림돌임을 확인.
    -   향후 대규모 피처 엔지니어링이나 외부 데이터 결합이 필요함.

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
```