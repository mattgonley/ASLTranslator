"""
:author: Ryan Nicholas, Matt Gonley, Marcus Tran, Nicole Fitz
:date: 1/31/2020
:description: This is our code for the training model for our neural network
"""
"""
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
import numpy as np
import matplotlib.pyplot as plt
import os
"""
from os import walk


class TrainingModel:

    def __init__(self, set_train, set_test):
        """
        Initialize Training Model
        :param set_test: test set
        :param set_train: training set
        """
        self.setTest = set_test
        self.setTrain = set_train
        print("Training: ", self.setTrain)
        print("Testing: ", self.setTest)

    def initialize_classification(self):
        """
        Initialize training model
        :return: None
        """
        f = []
        for (dirpath, dirnames, files) in walk(self.setTrain):
            f.extend(dirnames)
            break
        training_set = []
        for dirs in f:
            training_set.append([self.setTrain + "\\" + dirs, dirs])

