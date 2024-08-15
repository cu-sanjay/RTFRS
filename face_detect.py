import cv2 as cv
import sys

if len(sys.argv) != 2:
    print("Usage: python face_detect.py <image_path>")
    sys.exit()

image_path = sys.argv[1]
img = cv.imread(image_path)

if img is None:
    print(f"Error: Unable to load image at {image_path}")
    sys.exit()

cv.imshow('Image', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of faces found = {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected Faces', img)

cv.waitKey(0)
cv.destroyAllWindows()