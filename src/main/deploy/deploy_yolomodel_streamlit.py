import os

import streamlit as st
import cv2
from ultralytics import YOLO
import supervision as sv

def load_model(model_name):
    """
    Loads YOLOv8 custom model trained.
    :return: Trained YOLO model.
    """
    custom_model_path = os.path.join(os.path.dirname(__file__), '..', 'deploy', 'yolov8n_trained.pt')
    model_path = custom_model_path if model_name == 'custom' else "yolov8m.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    model = YOLO(model_path)
    if model_name != 'custom':
        model.fuse()  # Fuse the model for faster inference
    return model

def process_frame(frame, model):
    """
    Process a single frame using the YOLO model.
    :param frame: Input frame from the webcam.
    :param model: YOLO model.
    :return: Annotated frame.
    """
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    bounding_box_annotator = sv.BoundingBoxAnnotator()

    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)

    labeled_frame = label_annotator.annotate(
        scene=frame,
        detections=detections
    )

    annotated_frame = bounding_box_annotator.annotate(
        scene=labeled_frame,
        detections=detections
    )

    return annotated_frame

def main():
    st.title("YOLOv8 Object Detection with Streamlit")
    use_webcam = st.sidebar.checkbox("Use Webcam", value=True)
    model_name = st.sidebar.selectbox("Model Name", ["custom", "pretrained"], index=0)

    try:
        model = load_model(model_name)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    if use_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            annotated_frame = process_frame(frame, model)
            stframe.image(annotated_frame, channels="BGR")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
    else:
        st.write("Webcam is not enabled.")

if __name__ == '__main__':
    main()