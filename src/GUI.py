""""
:author: Ryan Nicholas, Matt Gonley, Marcus Tran, Nicole Fitz
:date: 1/31/2020
:description: Main file for the GUI, contains the window, camera input,
 and display of translated text
"""
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
from tkinter import *
import threading
import datetime
import imutils
import cv2
import os


class Window:

    def __init__(self, width, height, video_source = 0):
        """

        :param width: x "dimension" of the window,  pixels for the width
        :param height: y "dimension" of the window, pixels for the height of the window
        """
        
        self.root = Tk()
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.videoWidth = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.videoHeight = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.thread = None
        self.root.title("Sign Language Translator")
        geometry = "%dx%d" % (width, height)
        self.root.geometry(geometry)
        TextBox(self.root)
        self.root.mainloop()

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release
            self.window.mainloop()

