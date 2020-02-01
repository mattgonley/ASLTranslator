"""
:author: Ryan Nicholas, Matt Gonley, Marcus Tran, Nicole Fitz
:date: 1/31/2020
:description: This is our file for Camera Recognition

"""

import cv2 as cv
import numpy as np
import imutils

from __future__ import print_function
import argparse

class Camera:
    def __init__(self):
        self.x = -1
        self.y = -1
        cap = cv.VideoCapture(0)
        while (true)
            ret, frame = cap.read()

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            cv.imshow('frame', gray)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()


parser = argparse.ArgumentParser(description='Performs background subtraction')
parser.add_argument('--input', type= str, help='Path to a video or image',default=cap)
#will need to change default
parser.add_arguemnt('--algo', type=str, help='background subtraction method (KNN, MOG2).', default='MOG2')

args = parser.parse_args()
if args.algo == 'MOG2':
    backSub=cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(cv.samples.findFilesOrKeep(args.input))
if not capture.isOpened:
    print('Uable to open: ' + args.input)
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)


    cv.rectangle(frame, (10,2), (100, 20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)),
               (15,15), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0,))

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break