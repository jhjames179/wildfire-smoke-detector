from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"

st.set_page_config(page_title="산불 연기 탐지기", page_icon=":material/local_fire_department:", layout="wide")
st.title("산불 연기 탐지기")
st.write("학습된 YOLO 모델로 이미지에서 연기 후보 영역을 표시합니다.")

if not MODEL_PATH.exists():
    st.info("학습된 모델이 아직 없습니다. 데이터 준비와 학습은 README의 순서대로 실행하세요. 이 앱은 models/best.pt가 준비되면 탐지를 시작합니다.")

source = st.segmented_control("이미지 입력", ["사진 업로드", "카메라 촬영"], default="사진 업로드")
image_file = None
if source == "사진 업로드":
    image_file = st.file_uploader("연기 탐지할 이미지", type=["jpg", "jpeg", "png", "webp"])
else:
    image_file = st.camera_input("카메라로 장면 촬영")

confidence = st.slider("탐지 신뢰도 기준", min_value=0.05, max_value=0.95, value=0.25, step=0.05)

@st.cache_resource
def load_model(model_path: str, modified: float) -> YOLO:
    return YOLO(model_path)

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    left, right = st.columns(2)
    with left:
        st.image(image, caption="입력 이미지", width="stretch")

    if not MODEL_PATH.exists():
        st.warning("모델 파일이 없어 아직 탐지할 수 없습니다.")
    else:
        with st.spinner("연기 영역을 분석하고 있습니다..."):
            model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
            result = model.predict(np.asarray(image), conf=confidence, verbose=False)[0]
            rendered_bgr = result.plot()
            rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
        with right:
            st.image(rendered_rgb, caption="탐지 결과", width="stretch")
            st.download_button("결과 이미지 다운로드", data=cv2.imencode(".png", rendered_bgr)[1].tobytes(), file_name="smoke_detection.png", mime="image/png", icon=":material/download:")

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            st.warning("연기 후보를 찾지 못했습니다. 구름이나 안개 등 입력 이미지의 조건에 따라 결과가 달라질 수 있습니다.")
        else:
            st.success(f"연기 후보 {len(boxes)}개 탐지")
            rows = []
            for box in boxes:
                class_id = int(box.cls.item())
                rows.append({"클래스": model.names.get(class_id, str(class_id)), "신뢰도": round(float(box.conf.item()), 4)})
            st.dataframe(rows, hide_index=True)

st.caption("연구용 MVP입니다. 탐지 결과만으로 산불 여부를 확정하거나 긴급 대응을 결정하지 마세요.")

