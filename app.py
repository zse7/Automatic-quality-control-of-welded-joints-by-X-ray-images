import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
import tempfile
from PIL import Image
import os

class DefectDetectionPipeline:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def resize_with_padding(self, img, target_size=(640, 640)):
        h, w = img.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h))
        delta_w = target_size[1] - new_w
        delta_h = target_size[0] - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        if len(img.shape) == 2:  # Одноканальное
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)  # Преобразуем в RGB
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:  # Уже RGB
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded, scale, (left, top)

    def predict(self, image_path: str, conf_threshold: float = 0.25) -> list[dict]:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if image_path.endswith('.png') else cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        img, scale, offset = self.resize_with_padding(img) 

        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Изображение не в RGB: shape={img.shape}")

        results = self.model.predict(
            img,
            conf=conf_threshold,
            imgsz=640,
            verbose=False
        )

        detections = []
        for result in results:
            if result.masks is not None: 
                masks = result.masks.xy 
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                for mask, conf, class_id in zip(masks, confidences, class_ids):
                    polygon = mask / np.array([img.shape[1], img.shape[0]])  
                    polygon = polygon.flatten().tolist()
                    detections.append({
                        'class_id': int(class_id),
                        'class_name': self.model.names[class_id],
                        'confidence': float(conf),
                        'polygon': polygon 
                    })
            elif result.boxes is not None:  # Bounding boxes
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    detections.append({
                        'class_id': int(class_id),
                        'class_name': self.model.names[class_id],
                        'confidence': float(conf),
                        'bbox': {
                            'x1': float(box[0]),
                            'y1': float(box[1]),
                            'x2': float(box[2]),
                            'y2': float(box[3])
                        }
                    })

        return detections, img

    def visualize_results(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Визуализация полигонов и bounding boxes."""
        image_rgb = image.copy()
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            if 'polygon' in detection:  # Полигон
                points = detection['polygon']
                h, w = image.shape[:2]
                points = [(points[i] * w, points[i + 1] * h) for i in range(0, len(points), 2)]
                points = np.array(points, np.int32)
                cv2.polylines(image_rgb, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                x, y = points[0]
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image_rgb, (int(x), int(y) - label_size[1] - 10),
                             (int(x) + label_size[0], int(y)), (0, 255, 0), -1)
                cv2.putText(image_rgb, label, (int(x), int(y) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            elif 'bbox' in detection: 
                bbox = detection['bbox']
                x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image_rgb, (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(image_rgb, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return image_rgb

@st.cache_resource
def load_model():
    return DefectDetectionPipeline(model_path="models/best.pt")

def main():
    st.title("Детектор дефектов сварки")
    st.markdown("Загрузите изображение (.jpg или .png) для обнаружения дефектов")

    try:
        pipeline = load_model()
        st.success("Модель загружена!")
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return

    conf_threshold = st.sidebar.slider("Порог уверенности", 0.1, 1.0, 0.25, 0.05)

    uploaded_file = st.file_uploader("Выберите изображение", type=['jpg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_container_width=True)

        if st.button("Обнаружить дефекты"):
            with st.spinner("Обрабатываем..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg' if uploaded_file.name.endswith('.jpg') else '.png') as tmp_file:
                        image.save(tmp_file.name)
                        detections, processed_image = pipeline.predict(tmp_file.name, conf_threshold)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Результаты")
                        if detections:
                            st.success(f"Найдено дефектов: {len(detections)}")
                            for i, det in enumerate(detections, 1):
                                st.write(f"**Дефект #{i}:**")
                                st.write(f"- Класс: {det['class_name']}")
                                st.write(f"- Уверенность: {det['confidence']:.3f}")
                                if 'polygon' in det:
                                    st.write(f"- Полигон: {det['polygon']}")
                                else:
                                    st.write(f"- Координаты: ({det['bbox']['x1']:.1f}, {det['bbox']['y1']:.1f}) - ({det['bbox']['x2']:.1f}, {det['bbox']['y2']:.1f})")
                        else:
                            st.info("Дефекты не обнаружены")

                    with col2:
                        result_image = pipeline.visualize_results(processed_image, detections)
                        st.image(result_image, caption="Результат детекции", use_container_width=True)

                    os.unlink(tmp_file.name)
                except Exception as e:
                    st.error(f"Ошибка при обработке: {e}")

if __name__ == "__main__":
    main()