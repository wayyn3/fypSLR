import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('your_model.h5')

# Create a list of class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Open the webcam (you can change the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (32, 32))  # Resize to match the expected input shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame = np.expand_dims(frame, axis=-1)  # Add a channel dimension
    frame = frame.astype('float32') / 255.0

    # Predict the class label
    predictions = model.predict(np.expand_dims(frame, axis=0))
    predicted_class = class_labels[np.argmax(predictions)]

    # Display the frame with the predicted class label
    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-time Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
