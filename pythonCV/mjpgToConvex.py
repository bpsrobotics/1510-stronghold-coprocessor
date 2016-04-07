#!/usr/bin/env python2.7
import cv2
import numpy as np
# import sys

# Note: System arguments should take the form of an IP address of the video
# capture feed

# srcImg = cv2.VideoCapture()  # Define srcImg as image/video capture
#
# if len(sys.argv) != 2:
#     print("Error: specify an URL to connect to")
#     exit(0)
#
# url = sys.argv[1]
#
# srcImg.open("http://127.0.0.1:8080/stream.wmv")
# ret, frameImg = srcImg.read()  # Test
# imgY, imgX, imgChannels = frameImg.shape

srcImg = cv2.imread("/home/solomon/the-deal/RealFullField/19.jpg", 1)


def percentFromResolution(srcImg, yTargetRes, xTargetRes):
    imgY, imgX, imgChannels = srcImg.shape
    modPercentX = xTargetRes / imgX
    modPercentY = yTargetRes / imgY
    return [modPercentY, modPercentX]


def imgScale(toScale, percentX, percentY):
    scaledImg = cv2.resize(toScale, None, fx=percentX, fy=percentY,
                           interpolation=cv2.INTER_CUBIC)  # MaybeTry INTER_AREA
    return scaledImg


def threshHSL(imgSrc, lower, upper):
    """Returns binary mask of image based on HSL bounds"""
    imgSrcHLS = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2HLS)
    npLower = np.array([lower[0], lower[2], lower[1]])  # Compesate for HLSvsHSL
    npUpper = np.array([upper[0], upper[2], upper[1]])
    tmp = cv2.inRange(imgSrcHLS, npLower, npUpper)
    return tmp


def threshRGB(imgSrc, lower, upper):
    """Returns binary mask of image based on RGB bounds"""
    imgSrcRGB = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2RGB)
    npLower = np.array([lower[0], lower[1], lower[2]])
    npUpper = np.array([upper[0], upper[1], upper[2]])
    tmp = cv2.inRange(imgSrcRGB, npLower, npUpper)
    return tmp


def cvAdd(img1, img2):
    """Returns addition of 2 images"""
    tmp = cv2.add(img1, img2)
    return tmp


def findContours(img):
    """Finds contours in image, preferably binary image"""
    contours, hierarchy = \
        cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


a = threshHSL(srcImg, [50, 25, 34], [93, 255, 149])  # HSL thresh lower/upper
b = threshRGB(srcImg, [110, 119, 126], [255, 255, 255])  # RGB lower/upper
c = cvAdd(a, b)
d = c
contours, hiearchy = findContours(d)


tmpVar = 0

while len(contours) > 1:  # this inefficient mess finds the biggest contour
    # (I think)
    for z in range(0, len(contours)):
        try:
            if cv2.contourArea(contours[z]) <= tmpVar:
                contours.pop(z)
        except IndexError:
            break
        # print (str(tmpVar) + ": " + str(len(contours)) + ": " + str(z))
    tmpVar += 1


# rect = cv2.minAreaRect(contours[0])
# box = cv2.cv.BoxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(srcImg, [box], 0, (0, 255, 0), 2)
#
# rows, cols = srcImg.shape[:2]
# [vx, vy, x, y] = cv2.fitLine(contours[0], cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
# lefty = int((-x*vy/vx) + y)
# righty = int(((cols-x)*vy/vx)+y)
# cv2.line(srcImg, (cols-1, righty), (0, lefty), (255, 0, 0), 2)

hull = cv2.convexHull(contours[0], returnPoints=True)

(count,_,_) = hull.shape
hull.ravel()
hull.shape = (count, 2)

cv2.drawContours(srcImg, contours, -1, (0, 0, 255), 3)
# cv2.polylines(srcImg, np.int32([hull]), True, (0, 255, 0), 5)

tmpVar = 0
while len(cv2.approxPolyDP(contours[0], tmpVar, True)) != 8:
    if len(cv2.approxPolyDP(contours[0], tmpVar, True)) > 8:
        tmpVar += 1
    elif len(cv2.approxPolyDP(contours[0], tmpVar, True)) < 8:
        tmpVar -= 1

approx = cv2.approxPolyDP(contours[0], tmpVar, True)

print len(approx)


cv2.drawContours(srcImg, approx, -1, (0, 255, 0), 3)

cv2.imshow('e', srcImg)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
