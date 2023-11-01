import cv2
import numpy as np
import utils

webcam = False
path = './5.jpeg'

cap = cv2.VideoCapture(path)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)

scale = 3
wP = 297 * scale
hP = 210 * scale

while True:
    success, img = cap.read()
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    imgContours2, conts2 = utils.getContours(
        img, showCanny=True, minArea=30000, cThr=[60, 60], draw=True, filter=4)

    cv2.imshow('Warp', imgContours2)

    # cv2.imshow('Original', img)
    cv2.waitKey(0)
