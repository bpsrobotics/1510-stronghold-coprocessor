#!/usr/bin/env python3

import os
path = "/home/solomon/frc/the-deal/RealFullField/"
x = os.listdir(path)
for z in range(0, len(x)):
    x[z] = x[z].split('.')[0]

x.remove('')
x = sorted(x, key=int)

for n in x:
    os.system("./mjpgToConvex.py " + n)
    print (n)
