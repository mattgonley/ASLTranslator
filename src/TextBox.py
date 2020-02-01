import tkinter as tk
from tkinter import *
import cv2 as cv

class TextBox:
    def __init__(self):
        window = Tk()
        scrollbar = tk.Scrollbar(window)
        holder = tk.Text(window, height=50, width=500, yscrollcommand=scrollbar.set)
        holder.pack(side=tk.RIGHT)
        window.geometry("800x800")
        window.mainloop()
       # text.tag_configure('size', font=('Times New Roman', 14))
       # test = "Hello, World"
       # text.insert(test, 'size')






