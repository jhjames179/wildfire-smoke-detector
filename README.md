# Wildfire Smoke Detector

Pyro-SDIS의 산불 연기 바운딩박스로 YOLO 모델을 전이학습하고, Streamlit에서 사진 업로드 또는 카메라 촬영 이미지의 연기 후보를 표시하는 MVP입니다.

## 데이터셋 조사 및 선택

우선 사용할 데이터는 [Pyro-SDIS v2.0](https://huggingface.co/datasets/pyronear/pyro-sdis)입니다.

- 12,378장: train 11,249장, val 1,129장
- 연기 10,744장, 제공된 오탐/비연기 이미지 1,634장
- YOLO 형식 단일 클래스 `smoke` 바운딩박스. 빈 라벨은 원본에서 연기가 없다고 지정한 이미지입니다.
- Apache-2.0 라이선스. 데이터 카드의 버전 `v2.0`을 고정해 사용합니다.
- 분할 마스크와 별도 test split은 제공되지 않아 detection으로 학습합니다. 학습 과정에서 검증 세트 지표를 기록하며, 이를 test 성능으로 표현하지 않습니다.
- 비연기 1,634장은 구름/안개별 정답 클래스가 아닙니다. 각기 구름과 안개에 대한 성능은 별도 검증 이미지로 점검해야 합니다.

비교한 자료:

- [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset): 21,000장 이상, smoke/fire 박스 및 빈 라벨 포함, YOLO 포맷, OneDrive/Kaggle 다운로드. 단, 여러 원본 공개 데이터가 섞여 있어 이미지마다 원 라이선스 조건을 확인해야 합니다.
- [Boreal Forest Fire subset](https://etsin.fairdata.fi/dataset/1dce1023-493a-4d63-a906-f2a44f831898): CC BY 4.0, 연기 박스/세그멘테이션 자료 존재. 대용량(원본 4K 영상/이미지 포함)이고 자료 하위셋과 별도 다운로드 절차가 MVP 첫 데이터로는 무거워 제외했습니다.
- [SKLFS-WildFire Test](https://smoke.ustc.edu.cn/datasets/2/): 50,735장, 어려운 비연기 사례와 박스가 있지만 승인된 회원에게만 제공되어 즉시 다운로드 가능한 자료가 아닙니다.

## Windows 설치와 실행

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-training.txt
python data\download_pyro_sdis.py
python train.py
streamlit run app.py
```

데이터 다운로드는 약 1.32 GB입니다. 첫 학습에서는 Ultralytics가 사전학습 `yolo11n.pt` 가중치를 가져옵니다. 데이터셋과 가중치 다운로드 및 학습 시간은 인터넷 속도와 GPU/메모리에 따라 달라집니다. 기본값은 30 epoch, 640 입력, batch 8입니다. GPU 메모리가 부족하면 `train.py`의 `batch`를 줄이세요.

학습 완료 후 `models/best.pt`를 앱에서 사용합니다. Ultralytics 출력과 `runs/wildfire_smoke/pyro_sdis/`의 로그에 Precision, Recall, mAP가 저장되고, 학습 스크립트는 검증 지표를 출력합니다.

Streamlit Community Cloud에 배포할 때는 GitHub 저장소에 `app.py`, `requirements.txt`, `.streamlit/config.toml`, `models/best.pt`를 포함하고 `app.py`를 진입 파일로 선택합니다. 학습 데이터와 `runs/` 결과물은 배포 저장소에 넣지 않습니다.

## 현재 상태와 제한

- Pyro-SDIS Parquet 파일을 YOLO 형식으로 변환하고, train/val 사이 중복 이미지 159장을 제거했습니다. 최종 데이터는 train 11,109장, val 1,110장입니다.
- YOLO11n을 RTX 4060 Ti에서 30 epoch, 640 입력, batch 8로 학습했습니다. 가중치는 `models/best.pt`에 저장했습니다.
- 검증 결과: Precision 0.7609, Recall 0.6803, mAP50 0.7279, mAP50-95 0.3882. 이는 별도 test 세트가 아닌 Pyro-SDIS validation 세트 결과입니다.
- 검증 목록을 만들 때 같은 영상/카메라의 근접 프레임 누출 위험을 점검하고, 구름·안개·일반 풍경 테스트 이미지는 별도 출처와 사용 조건을 확인해 수집해야 합니다.
- 이 MVP는 이미지 단위 추론만 지원합니다. 실시간 스트리밍, 경보 전송, 사용자 정의 마스크는 포함하지 않습니다.
- 모델 출력은 연기 후보이며 산불 확정 판정이 아닙니다.
