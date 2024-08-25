# Real-Time Digit Detection with CNN
This project is a real-time digit detection application using a Convolutional Neural Network (CNN). The application captures images from a webcam, preprocesses them, and predicts the digit displayed in real-time.

# Screenshots
![Screenshot-2024-08-25-at-19 20 33-(2)](https://github.com/user-attachments/assets/73480896-ab7a-4400-b8a7-0d1cb2ee7e29)

## Overview

This project demonstrates the application of deep learning techniques to detect handwritten digits in real-time using a webcam. The CNN model is trained on a custom dataset of digit images, preprocessed and augmented for better accuracy. The application uses OpenCV for real-time image capture and preprocessing, TensorFlow and Keras for building and training the CNN model, and NumPy for numerical operations.


## Features

- **Real-Time Detection:** Captures webcam images and predicts digits in real-time.
- **Custom CNN Model:** Built using TensorFlow and Keras for accurate digit classification.
- **Image Preprocessing:** Applies grayscale conversion, histogram equalization, and normalization for better model performance.
- **Data Augmentation:** Uses various augmentation techniques to improve model generalization.
- **Easy Integration:** Simple setup and integration for real-time digit detection.


## Dataset

The dataset used contains tweets labeled as either positive or negative. The dataset is preprocessed to remove noise and prepare it for training the machine learning model.


## Training the Model

- Prepare your dataset: Organize your digit images into separate folders (0-9) under a main directory. Update the path variable in your script to point to this dataset directory.
- Train the model: Run the model_training.ipynb script to train the CNN model:
- This script will preprocess the images, augment the data, and train the model using TensorFlow and Keras.
- Save the model: After training, the model will be saved as model_trained.p. You can modify the script to change the model saving path.


## Testing with Webcam

- Run the real-time detection: Use the main.py script to start the webcam and test the real-time digit detection:
```bash
   python main.py
```
- Ensure that your webcam is properly connected and accessible. The application will display the processed image and predict the digit in real-time.
- Quit the application: Press q to exit the application.


## Dependencies

- Python 3.x
- OpenCV
- NumPy
- TensorFlow
- Keras
- Matplotlib
- Pickle
- Scikit-learn.


## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes.

