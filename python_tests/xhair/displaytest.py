#!/usr/bin/env python2
import numpy as np
import cv2
import os

cap = cv2.VideoCapture()
# Opening the link
cap.open("http://127.0.0.1:8080/stream.wmv")
ret, frameImg = cap.read()
imgY, imgX, imgChannels = frameImg.shape
posX = 0
posXmod = 1


def getRects(sourceImg, rectArray, rectTilt=0):
    """Overlays given rectangles on image."""

    # rectArray is a n-length array arrays (2d), like so:
    # [(1st x, 1st y), (2nd x, 2nd y), (r,g,b), thickness]
    for x in rectArray:
        cv2.rectangle(sourceImg, x[0], x[1], x[2], x[3])
    return sourceImg


def getLines(sourceImg, lineArray):
    """Overlays given lines on image."""
    # lineArray is a n-length array of arrays (2d or whatever), like so:
    # [(1st x, 1st y), (2nd x, 2nd y), (r,g,b), thickness]
    for x in lineArray:
        cv2.line(sourceImg, x[0], x[1], x[2], x[3])
    return sourceImg


def gridFill(sourceImg, xFreq, yFreq, thickness):
    # First the vertical lines (along x axis)
    imgY, imgX, imgD = sourceImg.shape
    for x in range(0, imgX):
        if x % xFreq == 0:
            cv2.line(sourceImg, (x, 0), (x, imgY), (255, 255, 255), thickness)
    # Now for y axis (horiz lines)
    for y in range(0, imgY):
        if y % yFreq == 0:
            cv2.line(sourceImg, (0, y), (imgX, y), (255, 255, 255), thickness)
    return sourceImg


def rectToLine(rect):
    """Turns rectangle (1x upper left coord, 1x lower right) to 4 lines"""
    # Give [(x1, y1), (x2, y2)]
    # Returns 4x of those
    x1, y1 = rect[0]
    x2, y2 = rect[1]
    returnArray = []
    returnArray.append([(x1, y1), (x1, y2)])
    returnArray.append([(x1, y1), (x2, y1)])
    returnArray.append([(x2, y2), (x1, y2)])
    returnArray.append([(x2, y2), (x2, y1)])
    return returnArray

rectArray = [
    # 1st vertex  2nd vertex  rgb color  thickness
    [(128*3 + 128/2, 72*3), (128*4 + 128/2, 72*7), (255, 0, 0), 3],
    [(128*4 + 128/2, 72*6), (128*5 + 128/2, 72*7), (255, 0, 0), 3],
    [(128*5 + 128/2, 72*3), (128*6 + 128/2, 72*7), (255, 0, 0), 3],
]

lineArray = [
    # [(450, 216), (0, 0), (255, 255, 255), 5]
]

for x in rectArray:
    for y in rectToLine([x[0], x[1]]):
        y.append((255, 0, 0))
        y.append(3)
        lineArray.append(y)
print (lineArray)

os.system('clear')
print (imgX)
print (imgY)
print (imgChannels)
print ("Press 'q' to exit")

# xHairImg = np.zeros((imgY, imgX, 3), np.uint8)
#
# # xHairImg = getRects(xHairImg, rectArray)
# xHairImg = getLines(xHairImg, lineArray)
#
# # Now convert xHairImg to 4 channel
#
# # cv2.imshow("raw", xHairImg)
#
# temp = cv2.cvtColor(xHairImg, cv2.COLOR_BGR2GRAY)
# retval, mask = cv2.threshold(temp, 10, 255, cv2.THRESH_BINARY)
#
# rgb = []
# rgb = cv2.split(xHairImg)
# print (mask.shape)
#
# rgba = [rgb[0], rgb[1], rgb[2], mask]
# xHairImgAlphaThresh = cv2.merge(rgba)

# print (xHairImgAlphaThresh.shape)
# cv2.imshow("mask", mask)
# cv2.imshow("thresholded", xHairImgAlphaThresh)

while True:
    # Capture frame-by-frame
    ret, frameImg = cap.read()
    # print ret
    # Display the resulting frame
    # frameImg = gridFill(frameImg, imgX/20, imgY/10, 1)
    # frameImg = getRects(frameImg, rectArray)
    frameImg = getLines(frameImg, lineArray)
    cv2.imshow('Mobile IP Camera', frameImg)
    # Clear screen
    # os.system('clear')
    # Exit key

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
