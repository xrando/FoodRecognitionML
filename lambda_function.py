import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='food_classification_model.tflite')
interpreter.allocate_tensors()

# Perform inference with the TensorFlow Lite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def lambda_handler(event, context):
    # Process the input image (event) and prepare it for inference
    input_data = preprocess_image(event['image'])
    input_data = input_data.astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-process the output and prepare the response
    result = {'food_probability': output_data[0][0], 'non_food_probability': 1 - output_data[0][0]}
    return result

def preprocess_image(image_data):
    # Add your image preprocessing code here (e.g., resizing, normalization)
    return processed_image
