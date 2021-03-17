import cv2
import numpy as np
import math
import numexpr as ne
import pprint
from matplotlib import pyplot as plt
import imutils
import sys
import linecache
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)

def showPlot(plt, silencePlots):
    if not silencePlots:
        plt.show()
    return

def showImg(winname, img, silencePlots):
	if not silencePlots:
		cv2.imshow(winname, img)
		cv2.waitKey(0)
	return

def showPrint(string, silencePrints):
    if not silencePrints:
        print(string)
    return

def interpAssay(assay, hsvFromPad):
	# - This function interpolates each assay
	assayCalibDict = {
		'urobilinogen': {
			'param': 'sat',
			'concRange': list(np.linspace(0,1000,100))
		},
		'glucose': {
			'param': 'hue',
			'concRange': list(np.linspace(0,1000,100))
		},
		'bilirubin': {
			'param': 'val',
			'concRange': list(np.linspace(0,3,100))
		},
		'ketones': {
			'param': 'sat',
			'concRange': list(np.linspace (0, 100, 100))
		},
		'specificGravity': {
			'param': 'hue',
			'concRange': list(np.linspace(1,1.03,100))
		},
		'blood': {
			'param': 'hue',
			'concRange': list(np.linspace(0, 250, 100))
		},
		'pH': {
			'param': 'hue',
			'concRange': list(np.linspace(5,9,100))
		},
		'protein': {
			'param': 'sat',
			'concRange': list(np.linspace(0, 1000,100)),
		},
		'nitrite': {
			'param': 'sat',
			'concRange': list(np.linspace(0, 2, 100))
		},
		'leukocytes': {
			'param': 'sat',
			'concRange': list(np.linspace(0, 500, 100))
		},
		'ascorbicAcid': {
			'param': 'sat',
			'concRange': list(np.linspace(0, 40,100))
		}
	}

	# + Run through same processing/scaling as calibration curve
	refColor = (0, 0, 0)

	# - Retrieve which HSV parameter to use
	targetDict = assayCalibDict[assay]
	hsvParam = targetDict['param']

	# + Scale appropriately
	if hsvParam == 'hue':
		hsvRaw = hsvFromPad[1]
		scaledHSVParam = min(abs(hsvRaw - refColor[0]), 255.0 - abs(hsvRaw - refColor[0])) / 128  # / 180.0
	elif hsvParam == 'sat':
		hsvRaw = hsvFromPad[2]
		scaledHSVParam = abs(hsvRaw - refColor[1]) / 255.0
	elif hsvParam == 'val':
		hsvRaw = hsvFromPad[3]
		scaledHSVParam = abs(hsvRaw - refColor[2]) / 255.0

	# + Interoplate parameter
	if assay == 'urobilinogen':
		assayResult = ((0.6640 * (np.arctan((0.1623/scaledHSVParam) + 0.3322))) + 2.3982) + (1.4109*scaledHSVParam)
	elif assay == 'glucose':
		assayResult = (((2705.6435 * (np.arctan((427290.0633/scaledHSVParam) + 1966.8450))) + -38771.1625) + (0.00028822*scaledHSVParam) + 34522.0699)
	elif assay == 'bilirubin':
		assayResult = 0.9397 + (-0.0426 * (scaledHSVParam ** 1.2008))
	elif assay == 'ketones':
		assayResult = 0.3435 + (0.0252 * (scaledHSVParam ** 0.4839))
	elif assay == 'specificGravity':
		assayResult = ((0.2188 * (np.arctan((684.0730/scaledHSVParam) + -679.1174))) + 50987.0780) + (-4.0613*scaledHSVParam) + -50982.3893
	elif assay == 'blood':
		assayResult = 0.2629 + (0.0255 * (scaledHSVParam ** 0.5965))
	elif assay == 'pH':
		assayResult = (1.0264 + ((-2.8194 - 1.0264) / (4.3717 + ((0.1410 * scaledHSVParam) ** 15.7145))))
	elif assay == 'protein':
		assayResult = ((1952.8780 * (np.arctan((404271.9767/scaledHSVParam) + 5184.5437))) + -2163.3303) + (-0.00001234*scaledHSVParam) + -903.6394
	elif assay == 'nitrite':
		assayResult = 0.3294 + (0.00000005 * (scaledHSVParam ** 22.4571))
	elif assay == 'leukocytes':
		assayResult = ((22.1788 * (np.arctan((6849.0873/scaledHSVParam) + 118.4425))) + 16850.7943) + (-0.00014901*scaledHSVParam) + -16884.7002
	elif assay == 'ascorbicAcid':
		assayResult = 0.54509804 + (0.00298474 * (scaledHSVParam ** 1.07400058))

	return assayResult

def showInMovedWindow(winname, img, x, y, silencePlots):
	# + Sometimes Python likes to open cv2 windows off-screen.
	cv2.namedWindow(winname)        # Create a named window
	cv2.moveWindow(winname, x, y)   # Move it to (x,y)
	showImg(winname, img, silencePlots)

	return

def detectArucoTags(queryImage, arucoDict, silencePlots):
	# + Finding aruco markers inside an image
	arucoDict = cv2.aruco.Dictionary_get(arucoDict)
	arucoParams = cv2.aruco.DetectorParameters_create()
	(corners, ids, rejected) = cv2.aruco.detectMarkers(queryImage, arucoDict, parameters=arucoParams)

	# - Optional resize and show
	image = imutils.resize(queryImage, height=800)
	showInMovedWindow('Image', image, 0, 0, silencePlots)

	return (corners, ids, rejected)


def four_point_transform(image, pts):
	# + Perform a perspective transform using four corner points
	(tl, tr, br, bl) = pts
	# - Compute the width of the new image, which will be the # maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# - Compute the height of the new image, which will be the maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# - construct the set of destination points to obtain a "birds eye view", (i.e. top-down view) of the image,
	# specifying points in the top-left, top-right, bottom-right, and bottom-left order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# + Compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# + Return the warped image
	return warped


def getContourStats(cnt):
	# - Get moments of contour
	M = cv2.moments(cnt)
	# - Get centroid of contour
	cx = int(M['m10'] / M['m00'])
	cy = int(M['m01'] / M['m00'])
	# - Get area of contour
	area = cv2.contourArea(cnt)

	return {'moments': M, 'centroid': (cx, cy), 'area': area}

def getArucoType(arucoName):
# define names of each possible ArUco tag OpenCV supports
	arucoDict = {
		"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
		"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
		"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
		"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
		"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
		"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
		"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
		"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
		"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
		"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
		"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
		"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
		"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
		"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
		"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
		"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
		"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
		"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
		"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
		"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
		"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
	}

	arucoType = arucoDict[arucoName]

	return arucoType

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    linenumber = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, linenumber, f.f_globals)
    exceptionString =  'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, linenumber, line.strip(), exc_obj)
    return exceptionString

def analyzeUrinalysis(queryFilename, tempFileName, resultStoragePath, silencePrints, silencePlots):
	# * General Algorithm workflow*
	# 1. Accept image as input
	# 2. Find fiducial markers on test housing
	# 3. Perform perspective transform
	# 4. Find locations of squares of each assay pad
	# 5. For each assay pad, find HSV color of circle within the test pad
	# 6. Using color calibrations for each assay, interpolate concentration
	# 7. Return JSON object with output results

	# + Initialize results dictionary
	resultsDict = {}

	# + Read image into cv2
	image = cv2.imread(queryFilename)

	# + Detect Aruco tags in image
	arucoType = getArucoType('DICT_4X4_50')
	(corners, ids, rejected) = detectArucoTags(image, arucoType, silencePlots)

	# + Initialize dictionary for centroid of arucos
	arucoCentroidDict = {}

	# + Find the centroids of each aruco tag and store in dictionary
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()
		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			# draw the bounding box of the ArUCo detection
			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			# compute and draw the center (x, y)-coordinates of the ArUco
			# marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			# draw the ArUco marker ID on the image
			cv2.putText(image, str(markerID),
						(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
						0.5, (0, 255, 0), 2)
			showPrint("[INFO] ArUco marker ID: {}".format(markerID), silencePrints)
			# show the output image

			smallImage = imutils.resize(image, height=800)

			# cv2.imshow("Image", smallImage)
			# showInMovedWindow('Image2', smallImage, 0, 0)
			#cv2.waitKey(0)

			arucoCentroidDict[markerID] = (cX, cY)
	else:
		showPrint("No Aruco detected", silencePrints)

	showPrint(arucoCentroidDict, silencePrints)


	# - Right now, only perform analysis if all four aruco's are detected
	# - I believe this can be trimmed down to three (reducing the number of images that gets returned):
	# 	If you know 3 corners of a parallelogram, the 4th can be found using trigonometry (opposite interior angles match)
	#   1) Find out which corner is missing.
	#   2) Find the corner furthest one from it - that is vertex of angle.
	#  	3) Map the lengths of the other sides of the parallelogram, from the vertex, using this angle
	if (len(arucoCentroidDict) == 4):

		# + Perform perspective transform using four centroids of aruco markers
		# Aruco locations
		# 1 = Top Left
		# 2 = Top Right
		# 3 = Bottom Left
		# 4 = Bottom Right
		ptsList = [arucoCentroidDict[1], arucoCentroidDict[2], arucoCentroidDict[4],
				   arucoCentroidDict[3]]
		ptsArray = np.array(ptsList, dtype='int64')
		warpedImg = four_point_transform(image, ptsArray)
		# - Optional show
		smallWarpedImg = imutils.resize(warpedImg, height=800)
		showInMovedWindow('Perspective Xform', smallWarpedImg, 0, 0, silencePlots)

		# + Create mask to isolate the center of the test strips
		firstThirdDist = math.floor(warpedImg.shape[1] * 0.4)
		lastThirdDist = math.floor(warpedImg.shape[1] * 0.6)
		mask = np.zeros(shape=warpedImg.shape, dtype=np.uint8)
		mask = cv2.rectangle(mask, (firstThirdDist,0), (lastThirdDist,warpedImg.shape[0]), (255,255,255), -1)
		# - Optional show
		smallMask = imutils.resize(mask, height=800)
		showInMovedWindow('Mask', smallMask, 0, 0, silencePlots)

		# + Apply the mask to the perpsective transformed image
		maskedImg = cv2.bitwise_and(warpedImg,mask)
		smallMaskedImg = imutils.resize(maskedImg, height=800)
		# - Optional show
		showInMovedWindow('Masked Img', smallMaskedImg, 0, 0, silencePlots)

		measurementTotImg = warpedImg.copy()

		# 7.5 mm + 5 + 2.5 = 15 mm to center of first square
		# total mask height should be 101.27
		# center of next square is 7 mm
		# total mask width = 29.62
		# half width = 29.62 /2
		yPadList = list(range(1,12))
		y0 = 15 # y-distance to center of first pad [=] mm
		maskHeight = 101.27 # [=] mm
		padYPctCoordsList = [((y0 + ((iY-1)*7))/101.27) for iY in yPadList]
		yPadHeightPctOfMask = 5 / 101.27
		padWidthPctOfMask = 5 / 29.62
		padXPctCoords = 0.5

		# + Put the measurement coordinates into scale for the image being analyzed
		# (working with pcts, not pix to avoid resolution problems)
		(imgHeight, imgWidth, imgChannels ) = measurementTotImg.shape
		yCentroidCoordList = [math.floor((yCPct*imgHeight)) for yCPct in padYPctCoordsList]
		# + cntsSorted is a misnomer -> this isn't sorted, just scaled
		cntsSorted = [(math.floor(imgWidth / 2), yCentroidPct) for yCentroidPct in yCentroidCoordList]

		# + Create storage dictionary for each assay
		# - The order of this list is critical: top to bottom of urinalysis strip
		assayList = ['urobilinogen', 'glucose', 'bilirubin', 'ketones', 'specificGravity', 'blood', 'pH', 'protein',
					 'nitrite','leukocytes', 'ascorbicAcid']
		measurementDict = {}

		# + Loop through assays
		for index, assay in enumerate(assayList):

			# + Get the locations of the assay we are working on
			showPrint(f'({index}, {assay}', silencePrints)
			iCentroid = cntsSorted[index]
			showPrint(f'Centroid {iCentroid}', silencePrints)

			# + Convert image to HSV space
			measurementImg = warpedImg.copy()
			measurementHSVImg = cv2.cvtColor(measurementImg, cv2.COLOR_BGR2HSV)

			# + Calculate how big of a circle we are going to use to make measurements
			measurementRadius = math.floor((padWidthPctOfMask * 0.25) * imgWidth)

			# + Make mask of measurement circle and calcualte average HSV
			circle_img = np.zeros((measurementHSVImg.shape[0], measurementHSVImg.shape[1]),np.uint8)
			cv2.circle(circle_img, iCentroid, measurementRadius, (255, 255, 255), -1)
			circleHSVMean = cv2.mean(measurementHSVImg, mask=circle_img)[::-1]
			cv2.circle(measurementTotImg, iCentroid, measurementRadius, (0, 255, 0), 3)
			# - Store result
			measurementDict[assay] = circleHSVMean


		# - Optional visualization of all measurement circles
		smallMeasurementTotImg = imutils.resize(measurementTotImg, height=800)
		showInMovedWindow('Measurements', smallMeasurementTotImg, 0, 0, silencePlots)
		showPrint(measurementDict, silencePrints)

		# + Create results dictionary for storage and return
		resultsDict = {}

		# + Send HSV measurements for each assay to be interpolated
		for assay in measurementDict:
			resultsDict[assay] = interpAssay(assay, measurementDict[assay])

		# + Set error flag as False
		resultsDict['errorFlag'] = 0
		showPrint(resultsDict,silencePrints)


	else:
		# + Error: Not enough Aruco tags found to do image processing
		resultsDict['errorFlag'] = -1
		resultsDict['errorMsg'] = f'Only {len(arucoCentroidDict)} Aruco found in image.'
		resultsDict['recommendation'] = 'Please take another photo.'

	# + Return results dictionary
	return resultsDict

