from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('food_classification_model.h5', compile=False)

@app.route('/')
def hello_world():
    return 'Hello from Flask!'

@app.route('/classify-image', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file_path = '/tmp/uploaded_image.jpg'
        file.save(file_path)
        img = image.load_img(file_path, target_size=(224, 224))
        os.remove(file_path)  # Remove the temporary file

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Perform prediction
        prediction = model.predict(img_array)
        result = "Non-Food" if prediction[0][0] >= 0.5 else "Food"

        return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
