#!/usr/bin/env python3

import sys
import time
from networktables import NetworkTable

import logging
logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) != 2:
    print("Error: specify an IP to connect to")
    exit(0)

ip = sys.argv[1]

NetworkTable.setIPAddress(ip)
NetworkTable.setClientMode()
NetworkTable.initialize()

ntTest = NetworkTable.getTable("SmartDashboard")

i = 0
while True:
    try:
        print('robotTime: ', ntTest.getNumber('robotTime'))
    except KeyError:
        print('robotTime: N/A')

    ntTest.putNumber('ntTest', i)
    time.sleep(1)
    i += 1
