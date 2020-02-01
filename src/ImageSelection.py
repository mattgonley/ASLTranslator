from __future__ import print_function

import cv2 as cv
import argparse
from src import TrainingModel
import os

def shape_selection(event, x, y, flags, param):
    global ref_point, cropping

    if event == cv.EVENT_LBUTTONDOWN:
        ref_point = [x, y]
        cropping = True
    elif event == cv.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        cv.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv.imshow("image", image)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the image")
arg = vars(parser.parse_args)

image = cv.imread(args["image"])
clone = image.copy
cv.namedWindow("image")
cv.setMouseCallback("image", shape_selection)
while True:
    cv.imshow("image", image)
    key = cv.waitKey(1) & 0xFF

    keyboard = cv.waitKey(27)
    if keyboard == ord('r'):
        image = clone.copy()
    elif keyboard == 27
        break


if len(ref_point) ==2:
    crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
    show('crop_img', crop_img)

    cv.waitKey(0)
cv2.destroyAllWindows()