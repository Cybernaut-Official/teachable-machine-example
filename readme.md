# README

## Image Classification using TensorFlow

This project demonstrates an image classification pipeline using a TensorFlow SavedModel. It classifies images into two categories: Dog and Cat.

### Prerequisites

- Python 3.x
- Virtualenv

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Cybernaut-Official/teachable-machine-example.git
   cd teachable-machine-example
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**

   ```bash
   pip install -r requirements.txt
   ```

### Project Structure

- `model.savedmodel/`: Directory containing the TensorFlow SavedModel.
- `th.jpeg`: Example image for testing.
- `main.py`: Main script to run the classification.

### Usage

1. **Load and preprocess the image**

   The `load_image` function reads the image from the given path using OpenCV and ensures it's loaded correctly.

   ```python
   image_path = './th.jpeg'
   image = load_image(image_path)
   ```

2. **Preprocess the image**

   The `preprocess_image` function resizes the image to 224x224 pixels, normalizes it, and adds a batch dimension.

   ```python
   preprocessed_image = preprocess_image(image)
   ```

3. **Load the model**

   The model is loaded using TensorFlow's `TFSMLayer`.

   ```python
   model_path = './model.savedmodel'
   model_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
   ```

4. **Make a prediction**

   The `get_prediction` function processes the image through the model, extracts the prediction, and returns the predicted label and confidence.

   ```python
   predicted_label, confidence = get_prediction(image)
   ```

5. **Print the prediction**

   The predicted label and confidence score are printed.

   ```python
   print(f'Predicted: {predicted_label} with confidence: {confidence:.2f}')
   ```

### Running the script

To run the script and classify an image:

```bash
python main.py
```

### Troubleshooting

- Ensure the image path is correct and the image can be loaded.
- Verify the model path and that the model is correctly saved in the `SavedModel` format.
- Check if TensorFlow and OpenCV are properly installed.

### License

This project is licensed under the MIT License.

---


