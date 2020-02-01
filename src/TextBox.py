import tkinter as tk
from tkinter import *
import cv2 as cv

class TextBox:
    def __init__(self):
        window = Tk()
        window.geometry("800x800")
        scrollbar = tk.Scrollbar(window)
        holder = tk.Text(window, height=50, width=500, yscrollcommand=scrollbar.set)
        holder.pack(side=tk.RIGHT)
        holder.tag_configure('size', font=('Times New Roman', 14))
        holder.insert('****', 'size')
        window.mainloop()
