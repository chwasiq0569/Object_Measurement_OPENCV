import cv2
import numpy as np
import utils

webcam = False
path = './1.jpeg'

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
    img, conts = utils.getContours(
        img, showCanny=True, minArea=50000, draw=True, filter=4)

    if len(conts) != 0:
        biggest = conts[0][2]
        # print(conts[0])
        imgWarp = utils.warpImg(img, biggest, wP, hP)
        cv2.imshow('Warp', imgWarp)

    cv2.imshow('Original', img)
    cv2.waitKey(0)
