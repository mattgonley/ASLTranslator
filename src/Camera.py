"""
:author: Ryan Nicholas, Matt Gonley, Marcus Tran, Nicole Fitz
:date: 1/31/2020
:description: This is our file for Camera Recognition

"""

import cv2 as cv
import numpy as np
import imutils


class Camera:
    def __init__(self):
        self.x = 0
        self.y = 0
