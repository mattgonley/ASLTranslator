import tkinter as tk
from tkinter import *
import cv2 as cv

class TextBox:
    def __init__(self):

        scrollbar = tk.Scrollbar(master=None)
        text = tk.Text(master=None, height=50, width=500, yscrollcommand=scrollbar.set)
        text.pack(side=tk.RIGHT)
        scrollbar.pack(side=RIGHT, fill=tk.Y)
        text.tag_configure('size', font=('Times New Roman', 14))
        test = "Hello, World"
        text.insert(test, 'size')






