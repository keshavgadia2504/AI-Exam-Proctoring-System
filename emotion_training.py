import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array

# Define dataset path and labels
dataset_path = "dataset"
emotions = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

# Load images and labels
data = []
labels = []

for i, emotion in enumerate(emotions):
    emotion_folder = os.path.join(dataset_path, emotion)
    
    for image_file in os.listdir(emotion_folder):
        img_path = os.path.join(emotion_folder, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))  # Resize to (48, 48)
        img = img_to_array(img) / 255.0  # Normalize
        
        data.append(img)
        labels.append(i)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# One-hot encoding of labels
labels = to_categorical(labels, num_classes=len(emotions))

# Split into training & testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Reshape input for CNN
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(emotions), activation='softmax')  # Output layer with 5 classes
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32)

# Save the trained model
model.save("emotion_model.h5")
print("Model training complete! Model saved as 'emotion_model.h5'.")
