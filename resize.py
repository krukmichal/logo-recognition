import cv2 as cv
import sys
img = cv.imread(cv.samples.findFile("./images/pobr-input5.jpg"))

scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
print(img.shape[1])
print(img.shape[0])
print(width)
print(height)
dim = (width, height)
  
# resize image
resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

if img is None:
    sys.exit("Could not read the image.")


#cv.imshow("Display window", resized)
cv.imwrite("./images/pobr-input5-a.jpg", resized)

k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("starry_night.png", img)

