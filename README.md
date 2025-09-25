# Image Classification with Neural Networks (CIFAR‑10)

A clean, beginner‑friendly CNN that trains on the CIFAR‑10 dataset using Keras/TensorFlow, with convenient command‑line flags to control training iterations, retraining behavior, and plotting. Includes a saved model artifact and quick image preview support.

## Highlights
- Trains a compact CNN on CIFAR‑10 (10 object categories)
- One‑line control of training iterations: `--epochs N`
- Skip retraining and auto‑load a saved model: `cifar10_model.h5`
- Force a fresh train any time: `--retrain`
- Optional plotting of sample images and previews: `--no-plots` to disable
- Minimal, readable code in a single script: `main.py`

## Tech Stack
- Python 3.9+ (tested with Python 3.13 in local runs)
- TensorFlow / Keras 3
- OpenCV (image I/O)
- Matplotlib (visualization)

## Project Structure
```
.
├── main.py                 # Train / load / evaluate the model; optional previews
├── cifar10_model.h5        # Saved model (created after first successful training)
├── 1.jpg 2.jpg 3.jpg 4.jpg # Optional local images for preview
└── README.md               # This file
```

## Setup
1) Create and activate a virtual environment (recommended)
- macOS / Linux (zsh)
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

2) Install dependencies
- Default (CPU):
```
pip install -r requirements.txt
```
- Apple Silicon (GPU acceleration, optional):
  - Replace `tensorflow` with `tensorflow-macos` and optionally add `tensorflow-metal`:
```
pip uninstall -y tensorflow && pip install tensorflow-macos tensorflow-metal matplotlib opencv-python numpy
```

## Usage
- Train (or load if already trained) with default 10 epochs and show plots:
```
python main.py
```
- Train for a custom number of iterations (epochs), e.g., 5:
```
python main.py --epochs 5
```
- Force a fresh retrain (ignores any existing `cifar10_model.h5`):
```
python main.py --retrain --epochs 15
```
- Run without any plot windows (handy for headless or quick runs):
```
python main.py --no-plots
```

Notes
- The script automatically trims the dataset for faster local training (20k train / 4k test). Adjust these slices in `main.py` if you want the full dataset.
- After training, the model is saved as `cifar10_model.h5` and will be reused on subsequent runs unless you pass `--retrain`.

## What You’ll See
- A 4×4 grid of CIFAR‑10 sample images (unless `--no-plots`)
- Training progress with validation metrics
- Final evaluation Loss/Accuracy
- If present, a preview of the first available image among `1.jpg`, `2.jpg`, `3.jpg`, `4.jpg`, or `wave.1.jpg` (unless `--no-plots`)

## Troubleshooting
- ValueError: “File format not supported … image_classfier.model”
  - Keras 3’s `load_model()` supports `.keras` and `.h5`. Use `cifar10_model.h5` (already wired) or resave your model with one of those extensions.
- AttributeError: “module 'keras.models' has no attribute 'save'”
  - Call `save()` on the model instance, e.g., `model.save('cifar10_model.h5')`. This project already does that.
- Matplotlib windows don’t show up
  - Use `--no-plots` to run without any GUI, or ensure you’re using a compatible backend.
- TensorFlow install on macOS
  - On Apple Silicon, prefer `tensorflow-macos` (and optionally `tensorflow-metal`). On Intel Macs or other platforms, use the standard `tensorflow` package.

## Roadmap / Ideas
- Add better data augmentation and regularization
- Use callbacks (EarlyStopping, ModelCheckpoint) to speed up and stabilize training
- Export to the native Keras v3 format (`.keras`) in addition to `.h5`
- Add a dedicated inference path to run predictions on custom images

Enjoy exploring and extending the model! If you’d like a prediction script for custom images, open an issue or request and we’ll add it.

