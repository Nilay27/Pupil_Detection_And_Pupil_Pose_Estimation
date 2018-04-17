import cv2
import math
import numpy as np;



# Read image (im=gray, img=colour)
im = cv2.imread('TestData/Real/02.jpg', 0)
img = cv2.imread('TestData/Real/02.jpg')

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 3000
params.maxArea = 6000
params.filterByCircularity = True
params.minCircularity = 0.5

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keyPoints = detector.detect(im)


for keypoint in keyPoints:
   x = int(keypoint.pt[0])
   y = int(keypoint.pt[1])
   s = keypoint.size
   r = int(math.floor(s/2))
   print x, y
   cv2.circle(img, (x, y), r, (255, 255, 0), 2)

cv2.imshow("blobOutput", img)
cv2.waitKey(0)