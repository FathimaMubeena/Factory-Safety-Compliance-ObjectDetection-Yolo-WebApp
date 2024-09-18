#Factory Safety Compliance Object Detection using YoloV10

## Project Purpose:
To demonstrate a full Computer Vision ML project from scratch to deployment to a web app usingthe following technology 
1. YoloV10
2. DeepLearning - CNN 
3. Streamlit

## Business Case:
The Factory Safety Regulations are to be followed by the Factory Employees in each location. This is causing a lot of accidents and the company is paying a lot of Medical Insurance to the employees. The company wants to detect if the Safety regulations are being followed by the Factory Employees in each location.
With this Computer Vision Deep Learning model we can detect if the Safety regulations are being followed by the Factory Employees in each location and this will decrease the liability and accident Medical insurance provided to the employees.

## Goal:
The goal of this project is to create a custom object detection model using YoloV10 to detect if the Safety regulations are being followed by the Factory Employees in each location.
And to accurately predict the objects in the image and classify them into the 5 classes - Helmet, Goggles, Jacket, Gloves, Footwear.

## Solution:
Steps to solve the problem:
Create a custom object detection model using YoloV10 to detect if the Safety regulations are being followed by the Factory Employees in each location.

1. Data Collection - Extraction
2. Data Preparation
3. Data annotation
4. Dataset Structure
5. Train YoloV10
6. Inference/Test YoloV10
7. Validate the model
8. Deploy model to Streamlit

Step 1 and 2 are already done for the factory safety equipment/object detection, we are not performing the tasks.
You can find the data in resources/[factory_safety_equipment_data](src%2Fmain%2Fresources%2Ffactory_safety_equipment_data)

There 5 objects/class that are labeled in the dataset. They are **_['Helmet', 'Goggles', 'Jacket', 'Gloves', 'Footwear']_**

If you still want to Annotate the Data - use the HumanSignal_labelImg , which is now [LabelStudio](https://github.com/HumanSignal/labelImg?tab=readme-ov-file#label-studio-is-a-modern-multi-modal-data-annotation-tool) - which is a modern Multi-modal data-annotation tool.

## Instructions to run the code
Steps:
1. Create a virtual environment:  
   ```shell
    python3 -m venv yolov8_env
    ```
2. Activate the virtual environment:  
      On macOS:
      ```shell
      source yolov8_env/bin/activate
      ```
3. Install the required packages:  
   ```shell
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
    ```shell
    streamlit run src/main/deploy/deploy_yolomodel_streamlit.py
    ```

## Results and Model Evaluation:

## Business Impact:
End users will be able to use the web app built off of this model to predict loan approvals right in front of the borrower. There will be no missed revenue opportunities since the model captures all true approvals (recall is 100%), and only a small portion of borrowers predicted to be approved will actually be denied. This will speed up the manual approval process and allow the company to process more loans in less time, resulting in more clients and revenue.


### Next Steps: Monitor performance and retrain model with more data as more data becomes available. and Deploy the model to a web app using Roboflow.
## Roboflow
Roboflow is a platform that provides tools and services for **building**, **training**, and **deploying** computer vision models. It offers features such as:  
1. **Dataset Management:** Tools for uploading, organizing, and preprocessing image datasets.
2. **Annotation Tools:** Integrated tools for annotating images with bounding boxes, segmentation masks, and more.
3. **Model Training:** Support for training models using popular frameworks like YOLO, TensorFlow, and PyTorch.
4. **Model Deployment:** Options for deploying trained models to various environments, including cloud services and edge devices.
5. **Inference API:** APIs for running inference on images using trained models.

Roboflow aims to simplify the end-to-end process of developing computer vision applications, from data collection and annotation to model training and deployment.
Some popular computer vision frameworks supported by Roboflow include:
1. YOLO (You Only Look Once)
2. TensorFlow
3. PyTorch

Roboflow provides a web-based interface for managing datasets, annotating images, training models, and deploying them to various platforms. It also offers APIs for integrating computer vision models into applications and workflows.
