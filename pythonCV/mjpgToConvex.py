#!/usr/bin/env python2.7
import cv2
import os
import numpy as np
import sys

os.system('')
# Note: System arguments should take the form of an IP address of the video
# capture feed

srcImg = cv2.VideoCapture()  # Define srcImg as image/video capture

if len(sys.argv) != 2:
    print("Error: specify an URL to connect to")
    exit(0)

url = sys.argv[1]

srcImg.open(url)
ret, frameImg = srcImg.read()  # Test


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
    imgSrcHSL = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2HSL)
    tmp = cv2.inRange(imgSrcHSL, lower, upper)
    return tmp


def threshRGB(imgSrc, lower, upper):
    """Returns binary mask of image based on RGB bounds"""
    imgSrcRGB = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2RGB)
    tmp = cv2.inRange(imgSrcRGB, lower, upper)
    return tmp


def cvAdd(img1, img2):
    """Returns addition of 2 images"""
    tmp = cv2.add(img1, img2)
    return tmp


def findContours(img):
    """Finds contours in image, preferably binary image"""
    img2, contours, hierarchy = \
        cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


x = percentFromResolution(ret, 640, 480)
print (x)
