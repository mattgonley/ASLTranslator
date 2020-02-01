"""
:author: Ryan Nicholas, Matt Gonley, Marcus Tran, Nicole Fitz
:date: 1/31/2020
:description: This is our file for Camera Recognition

"""
from __future__ import print_function

import cv2 as cv
import argparse

cap = cv.VideoCapture(0)
while (True):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #cv.imshow('frame', gray)
        parser = argparse.ArgumentParser(description='Performs background subtraction')
        parser.add_argument('--input', type=str, help='Path to a video or image', default=cap)
        parser.add_argument('--algo', type=str, help='background subtraction method (KNN, MOG2).', default='MOG2')
        args = parser.parse_args()
        if args.algo == 'MOG2':
            backSub = cv.createBackgroundSubtractorMOG2()
        else:
            backSub = cv.createBackgroundSubtractorKNN()
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            fgMask = backSub.apply(frame)
            cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
            cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)),
                       (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0,))
            cv.imshow('Frame', frame)
            cv.imshow('FG Mask', fgMask)
            keyboard = cv.waitKey(27)
            if keyboard == 27:
                exit(0)
                break
cap.release()
cv.destroyAllWindows()


