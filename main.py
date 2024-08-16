import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

width = 640
height = 480

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Camera not found or failed to open.")
    exit()

# Load the trained model
with open("model_trained_X.p", "rb") as f:
    model = pickle.load(f)

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOriginal = cap.read()
    if not success:
        print("Error: Failed to capture image.")
        break
    
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    

    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)
    print("Predicted class:", classIndex[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
