# OpenCV Face Recognition - Usage Guide

This guide provides instructions on how to use the various scripts included in the OpenCV Face Recognition project.

---

![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-blue.svg) ![Python](https://img.shields.io/badge/Python-3.12.0-brightgreen.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/cu-sanjay/RTFRS
cd RTFRS
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage Instructions

### 1. Train the Face Recognition Model

Train the model with images stored in the `Faces/train/` directory.

```bash
python faces_train.py
```

This command will create the following files:
- **face_trained.yml**
- **features.npy**
- **labels.npy**

### 2. Face Detection in a Static Image

Detect faces in an image file.

```bash
python face_detect.py <image_path>
```

- Replace `<image_path>` with the path to your image.
- Detected faces will be displayed with a bounding box.

### 3. Face Recognition in a Static Image

Recognize faces in an image file.

```bash
python face_recognition.py --input <image_path>
```

- Replace `<image_path>` with the path to your image.
- Identified faces will be labeled with the corresponding name.

### 4. Live Face Detection

Perform real-time face detection using your webcam.

```bash
python face_detect_live.py
```

- Faces will be detected and highlighted in real-time.
- Press **D** to stop and close the window.

### 5. Live Face Recognition

Perform real-time face recognition using your webcam.

```bash
python face_recognition_live.py
```

- Recognized faces will be labeled with names in real-time.
- Press **C** to capture the current frame and save it as **captured_face.png**.
- Press **Q** to stop and close the window.

---

## External Resources

- [OpenCV Documentation](https://docs.opencv.org/4.10.0/)
- [Python Official Documentation](https://docs.python.org/3/)

---

## License

This project is licensed under the [MIT License](LICENSE).