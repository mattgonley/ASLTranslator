"""
:author: Ryan Nicholas, Matt Gonley, Marcus Tran, Nicole Fitz
:date: 1/31/2020
:description: This is our main code for the Sign Language Translation App
"""

from src import TrainingModel as mod
from src import GUI
from src import TextBox
import os
from os import walk
import cv2
from keras_preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    """
    This is our main code here 
    """
    #textbox = TextBox.TextBox()


    directory = os.getcwd()
    index = directory.index("src")
    directory = directory[0:index] + "asl-alphabet\\"

    model = mod.TrainingModel(directory + "asl_alphabet_train\\asl_alphabet_train\\",
                              directory + "asl_alphabet_test\\asl_alphabet_test\\")

    #model.initialize_classification()

    test_image = cv2.imread(directory + "asl_alphabet_train\\asl_alphabet_train\\L\\L10.jpg")
    output = test_image.copy()
    test_image = cv2.resize(test_image, (200, 200))
    test_image = test_image.astype('float') / 255.0

    test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))

    f = []
    for (dirpath, dirnames, files) in walk(directory + "asl_alphabet_test\\asl_alphabet_test\\"):
        f.extend(dirnames)
        break

    predict = model.evaluate(test_image)

    i = predict.argmax(axis=1)[0]

    print("Predict: ", predict)

    label = f[i]

    print('Prediction: ', label)

    #window = GUI.Window(800, 600)