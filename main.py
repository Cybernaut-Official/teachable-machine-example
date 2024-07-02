import tensorflow as tf
import numpy as np
import cv2

# Load the SavedModel
model_path = r'./model.savedmodel'
try:
    model_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Labels for classification
labels = ['Dog', 'Cat']

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize the image to match the model input
    image = image.astype('float32') / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to get prediction
def get_prediction(image):
    preprocessed_image = preprocess_image(image)
    prediction_dict = model_layer(preprocessed_image)
    print(prediction_dict)

    # Print the keys in the prediction dictionary
    print("Prediction dictionary keys:", prediction_dict.keys())
    
    # Extract the prediction from the dictionary using the appropriate key
    # Assuming the key might be different, you need to check the keys printed
    prediction = prediction_dict[next(iter(prediction_dict))]
    
    # Convert the prediction tensor to a numpy array
    if tf.is_tensor(prediction):
        prediction = prediction.numpy()

    # Ensure prediction is a 2D array
    if len(prediction.shape) == 1:
        prediction = np.expand_dims(prediction, axis=0)

    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of the highest score
    confidence = np.max(prediction)  # Get the highest score
    
    # If the predicted_class is out of bounds or confidence is None, return None
    if predicted_class >= len(labels) or confidence is None:
        return None, None

    return labels[predicted_class], confidence

# Function to load and preprocess the image from a file
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    return image

if __name__ == "__main__":
    # Replace with the path to your image
    image_path = './th.jpeg'
    image = load_image(image_path)

    if image is not None:
        predicted_label, confidence = get_prediction(image)
        if predicted_label is not None and confidence is not None:
            print(f'Predicted: {predicted_label} with confidence: {confidence:.2f}')
        else:
            print('Prediction could not be made.')
    else:
        print('No image to predict.')
