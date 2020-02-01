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

    def __init__(self, width, height):
        """

        :param width: x "dimension" of the window,  pixels for the width
        :param height: y "dimension" of the window, pixels for the height of the window
        """
        
        self.root = Tk()
        self.thread = None
        self.stopEvent = None
        self.root.title("Sign Language Translator")
        geometry = "%dx%d" % (width, height)
        self.root.geometry(geometry)
        self.root.mainloop()

class TextBox:
    def __init__(self, window):

        scrollbar = Scrollbar(window)
        text = Text(window, height=50, width=500, yscrollcommand=scrollbar.set)
        text.pack(side=window.RIGHT)
        scrollbar.pack(side=RIGHT, fill=window.Y)
        text.tag_configure('size', font=('Times New Roman', 14))
        test = "Hello, World"
        text.insert(test, 'size')
