---

# Document Scanner

This Python script captures video from the default camera, applies image processing techniques to detect and scan documents, and saves the scanned images.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

Make sure you have Python installed on your system. You can check by running:

```bash
python --version
```

If OpenCV is not installed, the script will attempt to install it using pip:

```bash
pip install opencv-python
```

## Usage

Run the script using:

```bash
python DocumentScannerMain.py
```

The application will open a window displaying the video feed from the camera. Follow the on-screen instructions to capture and save scanned images.

## Features

- Adjust threshold levels using trackbars.
- Automatically installs OpenCV if not already installed.
- Supports switching between camera feed and pre-recorded video.

## Author

Om Prakash Singh

---
