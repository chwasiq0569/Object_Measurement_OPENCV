import cv2
import numpy as np


def getContours(img, cThr=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, [5, 5], 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)
    cv2.imshow('imgThre', imgThre)

    contours, hiearchy = cv2.findContours(
        imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    finalContours = []

    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            bbox = cv2.boundingRect(approx)

            if filter > 0:
                if (len(approx) == filter):
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])

    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)

    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 255, 0), 3)

    return img, finalContours


def reorder(mypoints):
    mypoints = mypoints.reshape((4, 2))
    # print(mypoints.shape)
    add = mypoints.sum(1)
    mypointsNew = np.zeros_like(mypoints)
    mypointsNew[0] = mypoints[np.argmin(add)]
    mypointsNew[3] = mypoints[np.argmax(add)]
    diff = np.diff(mypoints, axis=1)
    mypointsNew[1] = mypoints[np.argmin(diff)]
    mypointsNew[2] = mypoints[np.argmax(diff)]
    return mypointsNew


def warpImg(img, points, w, h):
    # print(points)
    # print(reorder(points))
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp
