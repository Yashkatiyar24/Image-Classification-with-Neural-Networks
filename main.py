import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import os

# Load CIFAR-10 dataset
(training_images, training_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize images
training_images = training_images / 255.0
test_images = test_images / 255.0

# Class names
class_names = ['surfing','rave rope','parrot','skateboard','guitar','cloud','cat','dog','car','airplane']

# Reduce dataset for quick training
training_images = training_images[:20000]
training_labels = training_labels[:20000]
test_images = test_images[:4000]
test_labels = test_labels[:4000]

# Build or load CNN model (resume training if saved model exists)
if os.path.exists("cifar10_model.h5"):
    print("Found cifar10_model.h5 — loading model and continuing training.")
    model = load_model("cifar10_model.h5")
else:
    print("No saved model found — building a new model and training from scratch.")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

# Compile & Train (continues training if a saved model was loaded)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10,
          validation_data=(test_images, test_labels))

# Save model after training
model.save("cifar10_model.h5")
print("Model saved to cifar10_model.h5")

# Test on a custom image
img = cv.imread("4.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_resized = cv.resize(img, (32, 32))  # resize to CIFAR input size
img_normalized = img_resized / 255.0

plt.imshow(img_resized)
plt.axis("off")

# Make prediction
prediction = model.predict(np.expand_dims(img_normalized, axis=0))
index = np.argmax(prediction)

print(f"This image is a {class_names[index]}")
plt.show()
