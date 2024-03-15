[//]: # (# FoodRecognitionML)

[//]: # ()
[//]: # (# Steps to deploy on aws ec2)

[//]: # (1. Create an EC2 instance)

[//]: # (2. Install git on the instance)

[//]: # (3. Clone the repository)

[//]: # (5. Install pip)

[//]: # (6. pip install tensorflow-cpu==2.7.0)

[//]: # (7. pip install protobuf==3.20)

[//]: # (8. pip install flask)

[//]: # (9. pip install pillow)

[//]: # (10. pip install -U flask-cors)

[//]: # ()
[//]: # (# Steps to run the application)

[//]: # (flask run --host=0.0.0.0)

[//]: # ()


# Thyme & Budget Recognition Service

This repository contains the source code for Thyme & Budget Recognition Service along with the instructions to train the model. Built with Python, TensorFlow, and the lightweight yet robust MobileNetV2 architecture, our model is designed to accurately classify food and non-food items.

## Dataset

The model is trained on the [Food-5K image dataset](https://www.kaggle.com/datasets/trolukovich/food5k-image-dataset) provided on Kaggle. The dataset consists of food and non-food images, divided into 3 sets - train, validation, and evaluation. Each set contains 2 categories - food and non_food, the training set contains 3000 images each while the other set consists of 1000 images.

## Prerequisites

- Python 3.7 or later
- TensorFlow 2.x
- pip

## Local Setup

1. Clone the repository: `git clone https://github.com/xrando/FoodRecognitionML.git`
2. Navigate to the project directory: `cd FoodRecognitionML`
3. Install the required Python packages: `pip install -r requirements.txt`

## Training the Model

The model training script is `Food_Recognition_model_training.py`. This script uses TensorFlow and Keras to train a binary classification model. The model is based on the MobileNetV2 architecture, a lightweight deep learning model designed for mobile and embedded vision applications.

The script uses transfer learning, where the MobileNetV2 model, pre-trained on the ImageNet dataset, is used as a feature extractor. The output of this base model is then passed through a custom classification head, which consists of a global average pooling layer, a dense layer with 1024 units and ReLU activation, a dropout layer for regularization, and a final dense layer with sigmoid activation for binary classification.

The model is trained using the Adam optimizer with a learning rate schedule, binary cross-entropy loss, and accuracy as the metric.

The model has already been pre-trained and is available in the repository. However, if you wish to train the model again, run the model training script: `python Food_Recognition_model_training.py`

This will train the model and save it as `food_classification_model.h5` in the project directory.

## Evaluating the Model

After training, the model is evaluated on a test set. The script prints a classification report and confusion matrix, and plots the training and validation accuracy over epochs to visualize the model's learning process.

## Model Performance and Tuning

The training process was carried out over 20 epochs. The model's performance improved over time, as evidenced by the decreasing loss and increasing accuracy on the training set as seen in Training.txt. The validation loss fluctuated, indicating that the model was learning to generalize to unseen data, but also that it was sensitive to the specific samples in the validation set. The highest validation accuracy achieved was 97.78% in the 15th epoch.

However, the model's performance on the validation set varied, with the accuracy ranging from 54.03% to 97.78%. After the final epoch, the model achieved a training accuracy of 99.29% and a validation accuracy of 94.56%. This indicates that the model was able to learn the training data well and also generalize to unseen data to a significant extent.

The model was then evaluated on the test set, achieving an accuracy of 92%. The precision and recall for the two classes were also high, indicating that the model was able to correctly identify the majority of samples in both classes. The confusion matrix further illustrates this, showing that the model made relatively few misclassifications.

The detailed performance metrics on the test set are as follows:

| Metric       | Class 0 | Class 1 |
|--------------|---------|---------|
| Precision    | 0.98    | 0.87    |
| Recall       | 0.85    | 0.98    |
| F1-Score     | 0.91    | 0.92    |
| Support      | 500     | 500     |

- Accuracy: 0.92
- Macro Average: 0.92
- Weighted Average: 0.92

## Confusion Matrix

The confusion matrix for the model on the test set is as follows:

|            | Predicted: 0 | Predicted: 1 |
|------------|--------------|--------------|
| Actual: 0  | 427          | 73           |
| Actual: 1  | 8            | 492          |

## Running the Application Locally

Start the Flask application: `flask run --host=0.0.0.0`

The application will be accessible at `http://localhost:5000`.

## Testing the Application locally

Run the testing script: `python Food_Recognition_testing.py`

This will classify the images in the `test_imgs` directory as food or non-food.

## Deploying to AWS EC2

1. Create an EC2 instance (Ubuntu Server 20.04 LTS is recommended).
2. Connect to the instance via SSH.
3. Update the package lists for upgrades and new package installations: `sudo apt update`
4. Install Python 3, pip, and git: `sudo apt install python3 python3-pip git`
5. Clone the repository: `git clone https://github.com/xrando/FoodRecognitionML.git`
6. Navigate to the project directory: `cd FoodRecognitionML`
7. Install the required Python packages: `pip3 install -r requirements.txt`
8. Start the Flask application: `flask run --host=0.0.0.0`

The application will be accessible at your EC2 instance's public IP address on port 5000.

## Testing the Deployed Application

To test the deployed application, you can send a POST request to the deployed endpoint using the provided script after modifying the endpoint to that of the deployed server: `python Send_Post_Request_To_Endpoint.py`

This script sends an image to the `/classify-image` endpoint of the deployed application and prints the response.

## Author
Benjamin Loh Choon How, 15/3/2024