# -*- coding: utf-8 -*-
import cv2
import numpy as np


GRID_WIDTH = 60
GRID_HEIGHT = 60

NUM_WIDTH = 30
NUM_HEIGHT = 30

N_MIN_ACTIVE_PIXELS = 45


def preprocess_grid(im_number):



    retVal, im_number_thresh = cv2.threshold(im_number, 150, 255, cv2.THRESH_BINARY)

    
    for i in range(im_number.shape[0]):
        for j in range(im_number.shape[1]):
            dist_center = np.sqrt(np.square(GRID_WIDTH // 2 - i) + np.square(GRID_HEIGHT // 2 - j))
            if dist_center > GRID_WIDTH // 2 - 2:
                im_number_thresh[i, j] = 0


    n_active_pixels = cv2.countNonZero(im_number_thresh)

    return [im_number_thresh, n_active_pixels]


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


def extract_number(im_number):

    #pre-processing of grid
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

            retVal, number = cv2.threshold(number, 50, 255, cv2.THRESH_BINARY)

            # reshape it to 1 dimension and return
            return True, number.reshape(1, NUM_WIDTH * NUM_HEIGHT)

    # if there is no number, return zeros in one dimension
    return False, np.zeros(shape=(1, NUM_WIDTH * NUM_HEIGHT))