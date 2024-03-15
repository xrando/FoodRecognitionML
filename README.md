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


# FoodRecognitionML

This repository contains the source code for the food recognition service for Thyme & Budget using machine learning. The service is built with Python, TensorFlow, and Flask.

## Dataset

This model is trained on the [Food-5K image dataset](https://www.kaggle.com/datasets/trolukovich/food5k-image-dataset) provided on Kaggle. The dataset consists of food and non-food images, divided into 3 sets - train, validation, and evaluation. Each set contains 2 categories - food and non_food, each with 500 images.

## Prerequisites

- Python 3.7 or later
- pip
- git

## Local Setup

1. Clone the repository: `git clone https://github.com/xrando/FoodRecognitionML.git`
2. Navigate to the project directory: `cd FoodRecognitionML`
3. Install the required Python packages: `pip install -r requirements.txt`

## Training the Model

The model has been pre-trained and is available in the repository. However, if you wish to train the model again, run the model training script: `python Food_Recognition_model_training.py`

This will train the model and save it as `food_classification_model.h5` in the project directory.

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

## Model Performance and Tuning

[To be filled by the user]