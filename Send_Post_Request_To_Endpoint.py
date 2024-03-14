import requests


# API Gateway URL format: https://{restapi_id}.execute-api.{region}.amazonaws.com/{stage}/{resource_path}
# api_gateway_url = f'https://f6lo8mmxc7.execute-api.ap-southeast-2.amazonaws.com/initial/FoodIdentificationService'
#api_gateway_url = "http://127.0.0.1:5000/classify-image"
api_gateway_url = "http://ec2-3-25-255-163.ap-southeast-2.compute.amazonaws.com:5000/classify-image"

# Path to the image file you want to send
image_path = 'test_imgs/0.jpg'

# Create a dictionary containing the file
files = {'file': open(image_path, 'rb')}

# Send a POST request to your Lambda function via API Gateway
response = requests.post(api_gateway_url, files=files)

# Print the response from your Lambda function
print(response.text)