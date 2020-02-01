"""
:author: Ryan Nicholas, Matt Gonley, Marcus Tran, Nicole Fitz
:date: 1/31/2020
:description: This is our file for Camera Recognition

"""
from __future__ import print_function

import cv2 as cv
import argparse
from src import TrainingModel
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


