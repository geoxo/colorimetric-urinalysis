import cv2
import numpy as np
import sys

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
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

# Aruco Paramaters
arucoIDs = [1,2,3,4]
arucoIDStrings = ['topLeft.jpeg','topRight.jpeg','bottomLeft.jpeg','bottomRight.jpeg']
arucoDictType = 'DICT_4X4_50'
arucoWidth = 500
arucoImgPath = './Aruco Tags/'

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[arucoDictType])


for (id,fileString) in zip(arucoIDs, arucoIDStrings):
	print("[INFO] generating ArUCo tag type '{}' with ID '{}'".format(
		arucoDictType, id))

	tag = np.zeros((arucoWidth, arucoWidth, 1), dtype="uint8")

	cv2.aruco.drawMarker(arucoDict, id, arucoWidth, tag, 1)
	# write the generated ArUCo tag to disk and then display it to our
	# screen

	# maskedImGray = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
	# blurredGrayImg = cv2.GaussianBlur(maskedImGray, (5, 5), 0)
	otsu_threshold, binaryImg = cv2.threshold(tag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	cv2.imwrite(arucoImgPath+fileString, binaryImg)
	cv2.imshow(f"ArUCo Tag: {id}", binaryImg)


	cv2.waitKey(0)
# cv2.destroyAllWindows()

# print("End")