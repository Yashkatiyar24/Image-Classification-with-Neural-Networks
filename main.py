import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
import argparse
import os

# Parse CLI args for epochs and retrain flag
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--no-plots", action="store_true")
args, _unknown = parser.parse_known_args()

# Load CIFAR-10 dataset
(training_images, training_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize images
training_images = training_images / 255.0
test_images = test_images / 255.0

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

# Plot first 16 images
if not args.no_plots:
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(training_images[i])
        plt.xlabel(class_names[training_labels[i][0]])
    plt.tight_layout()
    plt.show()


training_images = training_images[:20000]
training_labels = training_labels[:20000]
test_images = test_images[:4000]
test_labels = test_labels[:4000]

# Train or load model
model_path = 'cifar10_model.h5'
model = None

if os.path.exists(model_path) and not args.retrain:
    try:
        # Prefer loading existing model to save time
        model = models.load_model(model_path, compile=False)
        # Compile to ensure metrics/optimizer are defined for evaluate/predict
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(f"Loaded model from {model_path}.")
    except Exception as e:
        print(f"Failed to load {model_path} due to: {e}. Retraining...")

if model is None:
    # Build and train a simple CNN
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=args.epochs, validation_data=(test_images, test_labels))
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    # Save trained model
    model.save(model_path)
    print(f"Saved model to {model_path}.")

# Display an external image if present
if not args.no_plots:
    candidate_images = [
        '1.jpg', '2.jpg', '3.jpg', '4.jpg', 'wave.1.jpg'
    ]
    img_path = next((p for p in candidate_images if os.path.exists(p)), None)
    if img_path:
        img = cv.imread(img_path)
        if img is not None:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(img)
            plt.title(f"Preview: {img_path}")
            plt.axis('off')
            plt.show()
    else:
        print("No preview image found (looked for 1.jpg/2.jpg/3.jpg/4.jpg/wave.1.jpg).")

prediction = model.predict(np.array([training_images]))
index = np.argmax(prediction)
print(f"Prediction: {class_names[index]}")