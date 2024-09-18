import os
import cv2
import argparse
from ultralytics import YOLO

def main(input_path):
    # Initialize the YOLO model
    model = YOLO('yolov8n.pt')

    # Load image
    if os.path.isdir(input_path):
        images = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    else:
        images = [input_path]

    # Train the model
    model.train(
        data=f'{input_path}/data.yaml',
        epochs=5,
        imgsz=275,
        plots=True
    )

    # Save the model weights to the deploy directory
    deploy_dir = os.path.join(os.path.dirname(__file__), '..', 'deploy')
    os.makedirs(deploy_dir, exist_ok=True)
    model.save(os.path.join(deploy_dir, 'yolov8n_trained.pt'))

    # Load validation data
    val_data = f'{input_path}/valid'
    val_images = [os.path.join(val_data, f) for f in os.listdir(val_data) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Run inference on validation data

    # Initialize the confusion matrix
    y_true = []
    y_pred = []
    for image in val_images:
        # Load image
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model(img)
        print("Running inference and saving results")
        print(results)

        # Get the ground truth and predicted labels
        y_true.append(0)
        y_pred.append(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        help="path to a single image or image directory")
    args = parser.parse_args()
    main(args.input)