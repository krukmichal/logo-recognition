import cv2 as cv
import numpy as np
import sys

files = ["./images/pobr-input8-a.jpg", "./images/pobr-input7-a.jpg", "./images/pobr-input5-a.jpg", "./images/pobr-input2-a.jpg"]
#filename = "./images/pobr-input8-a.jpg"

def find_logo(file):
    img = cv.imread(cv.samples.findFile(file))
    if img is None:
        sys.exit("Could not read the image.")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret,thresh1 = cv.threshold(gray,200,255,cv.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh1 = cv.dilate(thresh1, kernel) 

    cv.imshow("Display window", thresh1)
    k = cv.waitKey(0)

def main():
    print(files[int(sys.argv[1])])
    find_logo(files[int(sys.argv[1])])


if __name__ == "__main__":
    main()

