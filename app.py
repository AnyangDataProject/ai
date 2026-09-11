from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = FastAPI()

# 학습된 YOLO 모델
model = YOLO("model/yolov8s_seed0_best.pt")


# RDD2022 클래스 → 로드센스 클래스
CLASS_MAPPING = {
    "longitudinal_crack": "crack",
    "transverse_crack": "crack",
    "alligator_crack": "crack",
    "pothole": "pothole",
}


@app.get("/")
def root():
    return {
        "message": "Anyang Project AI Server"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # 이미지 읽기
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    # YOLO 추론
    results = model.predict(
        source=image,
        conf=0.25
    )

    detections = []

    for result in results:

        # 탐지된 객체
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            original_class_name = model.names[class_id]

            # 로드센스 클래스명으로 변환
            class_name = CLASS_MAPPING.get(
                original_class_name
            )

            # 우리가 사용하는 클래스가 아니면 무시
            if class_name is None:
                continue

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "className": class_name,
                "confidence": confidence,
                "bboxX": x1,
                "bboxY": y1,
                "bboxWidth": x2 - x1,
                "bboxHeight": y2 - y1
            })

        # YOLO가 Bounding Box를 그린 결과 이미지
        plotted_image = result.plot()

    # numpy 배열 → JPEG
    from PIL import Image as PILImage

    result_image = PILImage.fromarray(
        plotted_image[:, :, ::-1]
    )

    buffer = io.BytesIO()

    result_image.save(
        buffer,
        format="JPEG"
    )

    # Base64 변환
    result_image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "detections": detections,
        "resultImage": result_image_base64
    }