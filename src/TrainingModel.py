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
        train_A_dir = os.path.join(self.setTest, 'A')
        train_B_dir = os.path.join(self.setTest, 'B')
        train_C_dir = os.path.join(self.setTest, 'C')
