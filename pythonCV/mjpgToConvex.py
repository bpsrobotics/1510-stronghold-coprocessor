#!/usr/bin/env python3
import cv2
import os
import numpy as np
import sys
import colorsys

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


def HSL2BGR(h, s, l):
    bgr = []
    tmp = colorsys.hls_to_rgb(h, l, s)
    rgb[0] = tmp[0]
    rgb[1] = tmp[1]
    rgb[2] = tmp[2]
    return bgr
