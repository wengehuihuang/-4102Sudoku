#COMP4102
#Zhangwen Yan
#101040231

# reference: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
# reference: https://web.stanford.edu/class/cs315b/assignment1.html
# reference: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
# reference: https://homepages.inf.ed.ac.uk/rbf/HIPR2/canny.htm
# reference: https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html
# reference: https://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square
# reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations

###############################################################################################
import numpy as np
import cv2
import math

# Final 540*540
# 540 = 60 * 9
# CAN = 1
IMAGE_LENGTH = 60
# GRID_WIDTH = 40
# GRID_HEIGHT = 40
SUDOKU_SIZE = 9
# N_MIN_ACTIVE_PIXELS = 30
SIZE_CHECKBOARD = IMAGE_LENGTH * SUDOKU_SIZE
# NUM_WIDTH = 20
# NUM_HEIGHT = 20
# N_MIN_ACTIVE_PIXELS = 30

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
		# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
		# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype="float32")
		# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
		# return the warped image
	return warped
	
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect
	#make a gaussian filter let graph and sigma as parameter
def gaussian_filter(image,sigma = 0):
	#kernel = gaussian_2d_kernel(sigma)
	kernel = np.array(gaussian_2d_kernel(sigma), dtype=np.float32 )
	kernel = kernel / kernel.sum()
	#get the size of kernel
	size = 2 * math.ceil(3 * sigma)+ 1
	row, col = image.shape
	filter_row, filter_col= kernel.shape
	img_height= img_width = int((size-1)/2)
	#ImagePart= np.zeros(shape=image.shape, dtype=np.float32)
	#make a new matrix for input image
	ImagePart = np.zeros((row + (2 * img_height), col + (2 * img_width)), dtype=np.float32)
	ImagePart[img_height:ImagePart.shape[0] - img_height, img_width:ImagePart.shape[1] - img_width] = image
	#Traverse all pixels and Multiply the original image and the kernel(convolution)
	for i in range(row):
		for j in range(col):
			#image[i,j] = int(np.sum(np.dot(kernel, ImagePart[i:i+filter_row,j:j+filter_col])))
			image[i,j] = abs(np.sum(kernel * ImagePart[i:i+filter_row,j:j+filter_col]))
	#return as a graph
	return image.astype(np.uint8)

	# build a 2D gaussian kernel by sigma
def gaussian_2d_kernel(sigma = 0):
	#get the size of kernel
	hsize = 2 * math.ceil(3 * sigma)+ 1
	hsize = int(hsize)
	#print("hsize",hsize)
	kernel = np.zeros([hsize,hsize])
	center = hsize//2
	# check sigma equal to 0
	if sigma == 0:
		sigma = ((hsize-1)*0.5 - 1)*0.3 + 0.8
	s = 2*(sigma**2)
	sum_val = 0
	for i in range(0,hsize):
		for j in range(0,hsize):
			x = i-center
			y = j-center
			kernel[i,j] = np.exp(-(x**2+y**2) / s)
			sum_val += kernel[i,j]
	sum_val = 1/sum_val
	return kernel*sum_val	
	
	#A sobel filter and let the gaussian graph as a parameter
def sobel_filter(image):
	height, weight = image.shape
	#sobel stand matrix
	sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
	sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
	#make empty matrix for both xdirection and ydirection
	Gx = np.zeros(image.shape, dtype=np.float32)
	Gy = np.zeros(image.shape, dtype=np.float32)
	dSobel = np.zeros((height,weight))
	#Traverse all pixels and Multiply the original image and the kernel(convolution)
	for i in range(height-2):
		for j in range(weight-2):
			Gx[i + 1, j + 1] = abs(np.sum(image[i:i + 3, j:j + 3] * sx))
			Gy[i + 1, j + 1] = abs(np.sum(image[i:i + 3, j:j + 3] * sy))
			dSobel[i+1, j+1] = (Gx[i+1, j+1]*Gx[i+1,j+1] + Gy[i+1, j+1]*Gy[i+1,j+1])**0.5
	#for NMS function, get the radian of every pixels.
	theta = np.arctan2(Gx, Gy)
	#cv2.imshow("sobel X.jpg",Gx)
	#cv2.imshow("sobel Y.jpg",Gy)
	#return graph and radian
	
	return Gx,Gy
	
	#Get rid of the borders to get a pure Sudoku picture
def border_extraction(image):
	#use the circle as a structured element for the closing operation
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
	# divide the gray image by the closed image
	div = np.float32(image) / close
	#Normalize the graph
	img_brightness_adjust = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
	#Adaptive threshold and binarize the result to a dark base graph
	img_thresh = cv2.adaptiveThreshold(img_brightness_adjust, 255,
									   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
									   cv2.THRESH_BINARY_INV, 11, 7)
	img, contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#set a area as 0
	max_area = 0
	biggest_contour = None
	#Iterate through all the contours found, and take the biggest one AKA the outermost silhouette of Sudoku 
	for i in contours:
		area = cv2.contourArea(i)
		if area > max_area:
			max_area = area
			biggest_contour = i
	#Use a black mask to cover the outside of the sudoku
	mask = np.zeros(img_brightness_adjust.shape, np.uint8)
	#White outlines are drawn on the completely black image and the inside of the outline is filled
	cv2.drawContours(mask, [biggest_contour], 0, 255, cv2.FILLED)
	#fill other place with black
	cv2.drawContours(mask, [biggest_contour], 0, 0, 2)
	image_with_mask = cv2.bitwise_and(img_brightness_adjust, mask)
	return image_with_mask, img_brightness_adjust
	
def detect_contoursX(image):
	#Detect edges in x direction and y direction using Sobel operator
	dx,dy  = sobel_filter(image)
	dx = cv2.convertScaleAbs(dx)
	cv2.normalize(dx, dx, 0, 255, cv2.NORM_MINMAX)
	ret, close = cv2.threshold(dx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
	close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernelx, iterations=1)
	binary, contour, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for i in contour:
		x, y, w, h = cv2.boundingRect(i)
		if h / w > 5:
			cv2.drawContours(close, [i], 0, 255, -1)
		else:
			cv2.drawContours(close, [i], 0, 0, -1)

	close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, None, iterations=2)
	closex = close.copy()
	return closex
	
def detect_contoursY(image):
	#Detect edges in x direction and y direction using Sobel operator
	dx,dy  = sobel_filter(image)
	dy = cv2.convertScaleAbs(dy)
	cv2.normalize(dy, dy, 0, 255, cv2.NORM_MINMAX)
	retVal, close = cv2.threshold(dy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
	close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernely)
	binary, contour, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for i in contour:
		x, y, w, h = cv2.boundingRect(i)
		if w / h > 5:
			cv2.drawContours(close, [i], 0, 255, -1)
		else:
			cv2.drawContours(close, [i], 0, 0, -1)

	close = cv2.morphologyEx(close, cv2.MORPH_DILATE, None, iterations=2)
	closey = close.copy()
	return closey
	
def checkerboard_maker(image,res):
	img_dots = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	binary, contour, hierarchy = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	centroids = []
	for i in contour:
		if cv2.contourArea(i) > 20:
			mom = cv2.moments(i)
			(x, y) = int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])
			cv2.circle(img_dots, (x, y), 4, (0, 255, 0), -1)
			centroids.append((x, y))
	centroids = np.array(centroids, dtype=np.float32)
	
	c = centroids.reshape((100, 2))
	c2 = c[np.argsort(c[:, 1])]
	b = np.vstack([c2[i * 10:(i + 1) * 10][np.argsort(c2[i * 10:(i + 1) * 10, 0])] for i in range(10)])
	bm = b.reshape((10, 10, 2))
	
	res2 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	output = np.zeros((450, 450, 3), np.uint8)
	for i, j in enumerate(b):
		ri = i // 10
		ci = i % 10
		if ci != 9 and ri != 9:
			src = bm[ri:ri + 2, ci:ci + 2, :].reshape((4, 2))
			dst = np.array([[ci * 50, ri * 50], [(ci + 1) * 50 - 1, ri * 50], [ci * 50, (ri + 1) * 50 - 1],
							[(ci + 1) * 50 - 1, (ri + 1) * 50 - 1]], np.float32)
			retval = cv2.getPerspectiveTransform(src, dst)
			warp = cv2.warpPerspective(res2, retval, (450, 450))
			output[ri * 50:(ri + 1) * 50 - 1, ci * 50:(ci + 1) * 50 - 1] = warp[ri * 50:(ri + 1) * 50 - 1,
																		   ci * 50:(ci + 1) * 50 - 1].copy()
	img_correct = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
	img_checkboard = cv2.adaptiveThreshold(img_correct, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 7)
	img_checkboard = cv2.resize(img_checkboard, (SIZE_CHECKBOARD, SIZE_CHECKBOARD), interpolation=cv2.INTER_LINEAR)
		
	return img_checkboard
		

def main():
	image = cv2.imread('test.png')
	ratio = image.shape[0] / 600.0
	orig = image.copy()
	image = cv2.resize(image, (int(image.shape[1] / ratio), 600), interpolation=cv2.INTER_CUBIC)
	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img_Blur = cv2.medianBlur(img_gray, 3)
	img_Blur = cv2.GaussianBlur(img_Blur, (3, 3), 0)
	#img_Blur = gaussian_filter(img_Blur,0)
	img_Edged = cv2.Canny(img_Blur, 75, 200)
	image_with_mask, img_brightness_adjust = border_extraction(img_Blur)
	closex = detect_contoursX(image_with_mask)
	closey = detect_contoursY(image_with_mask)
	resAND = cv2.bitwise_and(closex, closey)
	resOR = cv2.bitwise_or(closex, closey)
	img_dots = cv2.cvtColor(img_Blur, cv2.COLOR_GRAY2BGR)
	binary, contour, hierarchy = cv2.findContours(resAND, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	centroids = []
	for i in contour:
		if cv2.contourArea(i) > 20:
			mom = cv2.moments(i)
			(x, y) = int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])
			cv2.circle(img_dots, (x, y), 4, (0, 255, 0), -1)
			centroids.append((x, y))
	centroids = np.array(centroids, dtype=np.float32)
	if len(centroids) ==100:
		
		checkerboard = checkerboard_maker(img_brightness_adjust,resAND)
		result = cv2.resize(checkerboard, (540, 540), interpolation=cv2.INTER_AREA)
		cv2.imshow("checkerboard",result)
		cv2.imwrite("result.png",result)
		
	else:
		cnts = cv2.findContours(img_Edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnts = sorted(cnts[1], key=cv2.contourArea, reverse=True)[:5]
		screenCnt = None
		for c in cnts:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			if len(approx) == 4:
				screenCnt = approx
				break
		if screenCnt is None:
			print("NONE")
			sys.exit(-1)
		cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

		warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
		warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
		warped = cv2.threshold(warped, 0, 255, cv2.THRESH_OTSU)[1]
		
		img_checkboard = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)
		#img_checkboard = cv2.resize(warped, (SIZE_CHECKBOARD, SIZE_CHECKBOARD), interpolation=cv2.INTER_LINEAR)
		img_checkboard = cv2.resize(img_checkboard, (SIZE_CHECKBOARD, SIZE_CHECKBOARD), interpolation=cv2.INTER_LINEAR)

		result = cv2.resize(img_checkboard, (540, 540), interpolation=cv2.INTER_AREA)
		cv2.imshow("4", result)
		cv2.imwrite("result.png",result)
		

	
	cv2.imshow("image",image)
	#cv2.imshow("gray",img_gray)
	#cv2.imshow("Blur",img_Blur)
	#cv2.imshow("Brightness",img_brightness_adjust)
	#cv2.imshow("Thresh",img_thresh)
	#cv2.imshow("Mask",image_with_mask)
	#cv2.imshow("Edged",img_Edged)
	#cv2.imshow("closex",closex)
	#cv2.imshow("closey",closey)
	#cv2.imshow("resAND",resAND)
	#cv2.imshow("resOR",resOR)
	

	cv2.waitKey(0)
main()