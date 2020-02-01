"""
:author: Ryan Nicholas, Matt Gonley, Marcus Tran, Nicole Fitz
:date: 1/31/2020
:description: This is our main code for the Sign Language Translation App
"""

from src import TrainingModel as mod
import GUI
import os

if __name__ == '__main__':
    """
    This is our main code here 
    """
    directory = os.getcwd()
    index = directory.index("src")
    directory = directory[0:index] + "asl-alphabet\\"

    model = mod.TrainingModel(directory + "asl_alphabet_train\\asl_alphabet_train\\",
                              directory + "asl_alphabet_test\\asl_alphabet_test\\")

    model.initialize_classification()

    window = GUI.Window(800, 600)