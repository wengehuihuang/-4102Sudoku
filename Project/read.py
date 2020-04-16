
import cv2
import numpy as np

DEBUG = 1
#
GRID_WIDTH = 60
GRID_HEIGHT = 60
# 
NUM_WIDTH = 40
NUM_HEIGHT = 40
# 
SUDOKU_SIZE = 9
#
N_MIN_ACTIVE_PIXELS = 45

######################################################

def extract_number(im_number):

    [im_number_thresh, n_active_pixels] = preprocess_grid(im_number)

    # the number of active pixels of a grid must > threshold
    if n_active_pixels > N_MIN_ACTIVE_PIXELS:

        # find biggest bounding box of the number
        [x_b, y_b, w, h] = find_biggest_bounding_box(im_number_thresh)

        # calculate the distance from the center of the box to the center of the grid.
        cX = x_b + w // 2
        cY = y_b + h // 2
        d = np.sqrt(np.square(cX - GRID_WIDTH // 2) + np.square(cY - GRID_HEIGHT // 2))

        # the distance above must < threshold
        if d < GRID_WIDTH // 4:

            # extract the number from grid.
            number_roi = im_number[y_b:y_b + h, x_b:x_b + w]

            # expand number into a square, the side length is the maximum of number's width and height.
            h1, w1 = np.shape(number_roi)
            if h1 > w1:
                number = np.zeros(shape=(h1, h1))
                number[:, (h1 - w1) // 2:(h1 - w1) // 2 + w1] = number_roi
            else:
                number = np.zeros(shape=(w1, w1))
                number[(w1 - h1) // 2:(w1 - h1) // 2 + h1, :] = number_roi

            # resize the number into standard size
            number = cv2.resize(number, (NUM_WIDTH, NUM_HEIGHT), interpolation=cv2.INTER_LINEAR)
            number = number.astype(np.uint8)
            retVal, number = cv2.threshold(number, 50, 255, cv2.THRESH_BINARY)

            # reshape it to 1 dimension and return
            return True, number.reshape(1, NUM_WIDTH * NUM_HEIGHT)

    # if there is no number, return zeros in one dimension
    return False, np.zeros(shape=(1, NUM_WIDTH * NUM_HEIGHT))
	
	

def find_biggest_bounding_box(im_number_thresh):

    b, contour, hierarchy1 = cv2.findContours(im_number_thresh.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

    biggest_bound_rect = []
    bound_rect_max_size = 0
    for i in range(len(contour)):
        bound_rect = cv2.boundingRect(contour[i])
        size_bound_rect = bound_rect[2] * bound_rect[3]
        if size_bound_rect > bound_rect_max_size:
            bound_rect_max_size = size_bound_rect
            biggest_bound_rect = bound_rect

    x_b, y_b, w, h = biggest_bound_rect
    x_b = x_b - 1
    y_b = y_b - 1
    w = w + 2
    h = h + 2
    return [x_b, y_b, w, h]
	
def preprocess_grid(im_number):

    retVal, im_number_thresh = cv2.threshold(im_number, 150, 255, cv2.THRESH_BINARY)

    for i in range(im_number.shape[0]):
        for j in range(im_number.shape[1]):
            dist_center = np.sqrt(np.square(GRID_WIDTH // 2 - i) + np.square(GRID_HEIGHT // 2 - j))
            if dist_center > GRID_WIDTH // 2 - 2:
                im_number_thresh[i, j] = 0

    n_active_pixels = cv2.countNonZero(im_number_thresh)

    return [im_number_thresh, n_active_pixels]

def knn_ocr_normal(test):
    # train knn
    samples = np.load('samples.npy')
    labels = np.load('label.npy')

    knn = cv2.ml.KNearest_create()
    knn.train(samples, cv2.ml.ROW_SAMPLE, labels)

    ret, result, neighbours, dist = knn.findNearest(test, k=5)

    return result

#############################################################################################################################

#############################################################################################################################

sudoku = np.zeros(shape=(9 * 9, NUM_WIDTH * NUM_HEIGHT))

img_puzzle = cv2.imread('result.png',0)
img_puzzle = cv2.resize(img_puzzle,(540,540),interpolation=cv2.INTER_AREA)

indexes_numbers = []
for i in range(SUDOKU_SIZE):
    for j in range(SUDOKU_SIZE):
        number = img_puzzle[i * GRID_HEIGHT:(i + 1) * GRID_HEIGHT][:, j * GRID_WIDTH:(j + 1) * GRID_WIDTH]
        hasNumber, sudoku[i * 9 + j, :] = extract_number(number)
        if hasNumber:
            indexes_numbers.append(i * 9 + j)
			
print("There are", len(indexes_numbers), "numbers")
print(indexes_numbers)

##################################################################################

test = np.zeros(shape=(len(indexes_numbers), NUM_WIDTH * NUM_HEIGHT))
for num in range(len(indexes_numbers)):
    test[num] = sudoku[indexes_numbers[num]]
test = test.reshape(-1, NUM_WIDTH * NUM_HEIGHT).astype(np.float32)

result = knn_ocr_normal(test)


sudoku_puzzle = np.zeros(SUDOKU_SIZE * SUDOKU_SIZE)
print( "The Sudoku on the image is shown below: ")
for num in range(len(indexes_numbers)):
    sudoku_puzzle[indexes_numbers[num]] = result[num]
sudoku_puzzle = sudoku_puzzle.reshape((SUDOKU_SIZE, SUDOKU_SIZE)).astype(np.int32)
print(sudoku_puzzle)



