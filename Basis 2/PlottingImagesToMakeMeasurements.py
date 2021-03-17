import math
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy import signal, misc
import peakutils
import json
from statistics import mean
import sys
import pandas as pd
import colorsys

# + Filename to read
testFilename = './Test Images/colorChart1.jpg'

# + Reading image in
img = cv2.imread(testFilename, cv2.IMREAD_COLOR)

# + Convert image to HSV
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# + Display HSV image
plt.figure()
plt.title("HSV Image")
plt.imshow(imgHSV, interpolation='bicubic')
plt.show()




