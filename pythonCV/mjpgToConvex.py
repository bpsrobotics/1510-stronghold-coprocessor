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
    img2, contours, hierarchy = \
        cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


x = threshHSL(srcImg, [50, 25, 34], [93, 255, 149])  # HSL thresh lower/upper
y = threshRGB(srcImg, [110, 119, 126], [255, 255, 255])  # RGB lower/upper
z = cvAdd(x, y)
cv2.imshow('image', z)
cv2.waitKey(0)
cv2.destroyAllWindows()
