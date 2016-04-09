#!/usr/bin/env python3
import pickle

import sys
from networktables import NetworkTable

import logging
logging.basicConfig(level=logging.DEBUG)

ip = "255.255.255.0"
ntName = "AutoAim"
serialFile = "/home/solomon/pickle.txt"

# def printUsage():
#     str = \
#         "Usage: ./NTSendArray.py <ip> <TableName> <PickleFile>"
#     print (str)
#
# try:
#     ip = sys.argv[1]
# except IndexError:
#     print("Error: specify an IP to connect to")
#     printUsage()
#     exit(0)
#
# try:
#     ntName = sys.argv[2]
# except IndexError:
#     print("Error: specify a NetworkTable to connect to")
#     printUsage()
#     exit(0)
#
# try:
#     serialFile = sys.argv[3]
# except IndexError:
#     print("Error: specify a file to read seralized data from")
#     printUsage()
#     exit(0)
# print (serialFile)

with open(serialFile, 'rb') as f:
    aeee = pickle.load(f)

inputList = aeee
# print (inputList)
# for x in inputList:
#     print (str(x) + ": " + str(inputList[x]))
NetworkTable.setIPAddress(ip)
NetworkTable.setClientMode()
NetworkTable.initialize()

ntTable = NetworkTable.getTable(ntName)


def pad(leng):
    x = ""
    for z in range(0, 15-leng):
        x += " "
    return x

for x in inputList:
    if (x != "p1") and (x != "p2") and (x != "p3") and (x != "p4"):
        ntTable.putNumber(str(x), inputList[x])
        print("Pushing " + pad(len(str(x))) + str(x) + ": " + str(inputList[x]))


for x in range(1, 5):
    for z in range(0, 2):
        ntTable.putNumber("p" + str(x) + chr(z + 120),
                          inputList["p" + str(x)][z])
        print("Pushing             p" + str(x) + chr(z + 120) +
              ": " + str(inputList["p" + str(x)][z]))
