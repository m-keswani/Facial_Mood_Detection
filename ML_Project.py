#Import neccessary lib...
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Function to load the FER2013 dataset
def load_fer2013_dataset(subsample_fraction=0.5):
    # Change the path to the actual path where you've extracted the dataset
    data = pd.read_csv(r"D:\ML Project\fer2013.csv")
    
    # Subsample the data
    data_subset = data.sample(frac=subsample_fraction, random_state=42)
    
    pixels = data_subset['pixels'].tolist()
    X = np.array([np.fromstring(pixel, dtype=int, sep=' ') for pixel in pixels])
    y = data_subset['emotion'].values
    return X, y

# Load facial expression dataset
print('here')
X, y = load_fer2013_dataset(subsample_fraction=0.1)
print(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X Train : -", X_train)

# Extract facial features (you might want to use more sophisticated feature extraction methods)
# For simplicity, let's assume you have flattened pixel values as features
X_train = X_train / 255.0  # Normalize pixel values to [0, 1]
X_test = X_test / 255.0

# Train an SVM classifier for expression classification
svm_classifier = SVC(kernel='rbf', C=10.0) # Radial Basis Function (rbf)
print("SVM classifier :-", svm_classifier)
svm_classifier.fit(X_train, y_train)
print('after svm fit...')

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)

# Now, let's perform facial detection on an example image using OpenCV
image_path = r"D:\ML Project\img3.png"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use a pre-trained face detector (e.g., Haarcascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
gray_image = gray_image / 255.0

# Iterate over detected faces and classify expressions using the trained SVM
for (x, y, w, h) in faces:
    face_roi = gray_image[y:y + h, x:x + w]

    # Flatten the face_roi into a 1D array
    face_roi_features = face_roi.flatten().reshape(1, -1)

    # Ensure that the number of features matches the training data
    if face_roi_features.shape[1] != X_train.shape[1]:
        # If not, reshape to match the expected number of features
        print("X_train shape:", X_train.shape)
        face_roi_features = face_roi_features.reshape(1, -1)[:, :X_train.shape[1]]

    # Predict the expression using the trained SVM
    # Define a mapping from label numbers to expression names
    expression_mapping = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Neutral"}

    # Predict the expression using the trained SVM
    expression_prediction = svm_classifier.predict(face_roi_features)

    # Draw rectangle around the face and put text indicating the predicted expression
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    predicted_expression_name = expression_mapping.get(expression_prediction[0], "Unknown")
    cv2.putText(image, f"Expression: {predicted_expression_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2)

# Display the image with detected faces and expressions
cv2.imshow("Facial Detection and Expression Classification", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Expression Prediction:", expression_prediction)
print("Predicted Class Distribution:", np.bincount(expression_prediction))
print("Face ROI Features:", face_roi_features)
