""""
:author: Ryan Nicholas, Matt Gonley, Marcus Tran, Nicole Fitz
:date: 1/31/2020
:description: Main file for the GUI, contains the window, camera input,
 and display of translated text
"""
from __future__ import print_function   


import PIL
from PIL import Image,ImageTk
import pytesseract
import cv2
from tkinter import *

width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = Label(root)
lmain.grid(row=1, column=1, sticky=N+W)

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()


holder = Text(root, height=25, width=25, font=('Times New Roman', 28), bg="black", fg="white")
scrollbar = Scrollbar(root, command=holder.yview)
holder['yscrollcommand'] = scrollbar.set
#scrollbar.config(command=holder.yview)
scrollbar.grid(row=1,column=6, sticky=N+S+W)
holder.grid(row=1, column=5)




root.configure(bg='black')
root.mainloop()