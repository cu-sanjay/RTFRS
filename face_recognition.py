import cv2 as cv
import numpy as np
import os
import argparse

# command line arguments
parser = argparse.ArgumentParser(description='Face recognition script.')
parser.add_argument('--input', type=str, required=True, help='Path to the input image')
args = parser.parse_args()

DIR = 'Faces/train'
people = [i for i in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, i))]

haar_cascade = cv.CascadeClassifier('haar_face.xml')
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# read input image
img_path = args.input
img = cv.imread(img_path)

if img is None:
    print(f"Error: Unable to load image at {img_path}")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Unidentified Person', gray)

# detect faces
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y + h, x:x + w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence:.2f}')

    cv.putText(img, f'{people[label]}', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)
cv.destroyAllWindows()