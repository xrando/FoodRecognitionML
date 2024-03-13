import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('food_classification_model.h5')

# Path to the test images folder
test_imgs_dir = os.path.join(os.getcwd(), 'test_imgs')

# Initialize counts
food_count = 0
non_food_count = 0

# Iterate over each image in the folder
for img_name in os.listdir(test_imgs_dir):
    img_path = os.path.join(test_imgs_dir, img_name)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Perform prediction
    prediction = model.predict(img_array)
    if prediction[0][0] >= 0.5:
        print(f"{img_name}: Non-Food")
        non_food_count += 1
    else:
        print(f"{img_name}: Food")
        food_count += 1

# Print counts
print(f"Food Count: {food_count}")
print(f"Non-Food Count: {non_food_count}")
