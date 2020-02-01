"""
:author: Ryan Nicholas, Matt Gonley, Marcus Tran, Nicole Fitz
:date: 1/31/2020
:description: This is our code for the training model for our neural network
"""

import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join


class TrainingModel:

    def __init__(self, set_train, set_test):
        """
        Initialize Training Model
        :param set_test: test set
        :param set_train: training set
        """
        self.setTest = set_test
        self.setTrain = set_train

    def initialize_classification(self):
        """
        Initialize training model
        :return: None
        """
        file_names = [f for f in listdir(self.setTrain) if isfile(join(self.setTrain, f))]
        training_data = []
        for files in file_names:
            print(files)
            training_data.append([self.setTrain + "\\" + files, files])
