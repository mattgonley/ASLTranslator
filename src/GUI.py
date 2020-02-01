""""
:author: Ryan Nicholas, Matt Gonley, Marcus Tran, Nicole Fitz
:date: 1/31/2020
:description: Main file for the GUI, contains the window, camera input,
 and display of translated text
"""
from __future__ import print_function

import tkinter

import PIL
from PIL import Image
from PIL import ImageTk
from tkinter import *
import threading
import datetime
import imutils
import cv2
import os


class Window:

    def __init__(self, video_source=0):
        """

        :param width: x "dimension" of the window,  pixels for the width
        :param height: y "dimension" of the window, pixels for the height of the window
        """
        
        self.root = Tk()
        self.vid = cv2.VideoCapture(video_source)

        self.canvas = tkinter.Canvas(self.root)
        self.canvas.pack()
        self.delay = 15
        self.update()
        self.root.title("Sign Language Translator")

        self.root.mainloop()


    def update(self):
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTK.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=Tk.NW)

        self.window.after(self.delay, self.update)

