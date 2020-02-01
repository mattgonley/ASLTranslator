"""
:author: Ryan Nicholas, Matt Gonley, Marcus Tran, Nicole Fitz
:date: 1/31/2020
:description: This is our file for Camera Recognition

"""
from __future__ import print_function

import cv2 as cv
import argparse
from src import TrainingModel
import numpy as np
import os

def classify_image(img):
    """
    Fix the image to be sent to the classifier
    :param img: frame for the image
    :return: image type
    """

    directory = os.getcwd()
    index = directory.index("src")
    directory = directory[0:index] + "asl-alphabet\\"
    model = TrainingModel.TrainingModel(directory + "asl_alphabet_train\\asl_alphabet_train\\",
                              directory + "asl_alphabet_test\\asl_alphabet_test\\")

    output = img.copy()
    test_image = cv.resize(img, (200, 200))
    test_image = test_image.astype('float') / 255.0

    test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))

    return model.evaluate(test_image)

recapture = 0
cap = cv.VideoCapture(0)

ret,frame = cap.read()
# setup initial location of window

ret, frame = cap.read()
x, y, w, h = 300, 200, 100, 50  # simply hardcoded the values
track_window = (x, y, w, h)
# set up the ROI for tracking
roi = frame[y:y + h, x:x + w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
while (True):
        parser = argparse.ArgumentParser(description='Performs background subtraction')
        #default is the video capture
        parser.add_argument('--input', type=str, help='Path to a video or image', default=cap)
        #default will be the KNN subtraction method, KNN seems to work better then MOG2
        parser.add_argument('--algo', type=str, help='background subtraction method (KNN, MOG2).', default='KNN')
        args = parser.parse_args()
        if args.algo == 'KNN':
            backSub = cv.createBackgroundSubtractorKNN()
        else:
            backSub = cv.createBackgroundSubtractorMOG2()

        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            fgMask = backSub.apply(frame)
            if ret == True:
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                # apply camshift to get the new location
                ret, track_window = cv.CamShift(dst, track_window, term_crit)
                # Draw it on image
                pts = cv.boxPoints(ret)
                pts = np.int0(pts)
                img2 = cv.polylines(frame, [pts], True, 255, 2)
                #cv.imshow('img2', img2)
            #shows the coloured frame
            cv.imshow('Frame', frame)

            #shows the masked frame
            cv.imshow('FG Mask', fgMask)
            # captures the frames
            cv.imwrite('frame.jpg', frame)
            if recapture == 0:
                print("Classifier: ", classify_image(frame))

            recapture = (recapture + 1) % 50

            keyboard = cv.waitKey(27)
            if keyboard == 27:
                exit(0)
                break
cap.release()
cv.destroyAllWindows()


