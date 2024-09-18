import argparse
import os
import cv2
from ultralytics import YOLO
import supervision as sv


def load_model(model_name):
    """
      Loads YoloV8 custom model trained.
        :return: Trained Yolo model.
    """
    if model_name == 'custom':
        model = YOLO("yolov8n_trained.pt")  # Load a custom trained model
    else:
        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8n model
        model.fuse() # Fuse the model for faster inference
    return model

def main(use_webcam, model_name):
    cap = cv2.VideoCapture(0)
    model = load_model(model_name)

    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    bounding_box_annotator = sv.BoundingBoxAnnotator()

    while True:
        if use_webcam:
            ret, frame = cap.read()
        else:
            test_dir = os.path.join(os.path.dirname(__file__), '../', 'resources')
            frame = cv2.imread(os.path.join(test_dir, 'factory_safety_equipment_data/valid/images/1.jpeg'))

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

        cv2.imshow('frame', annotated_frame)

        if cv2.waitKey(30) == 27:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_webcam", type=bool, default=True)
    parser.add_argument("--model_name", type=str, default="custom")
    args = parser.parse_args()
    main(args.use_webcam, args.model_name)