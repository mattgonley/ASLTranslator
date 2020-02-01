"""
:author: Ryan Nicholas, Matt Gonley, Marcus Tran, Nicole Fitz
:date: 1/31/2020
:description: This is our code for the training model for our neural network
"""

import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
import os
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
        self.sample_training_images = None
        self.model = None
        self.total_train = 0
        self.total_validate = 0
        self.batch_size = 200
        self.IMG_HEIGHT = 200
        self.IMG_WIDTH = 200
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
        training_images = []
        training_labels = []
        for dirs in f:
            training_images.append(self.setTrain + "\\" + dirs)
            training_labels.append(dirs)
            self.total_train += len(os.listdir(self.setTrain + "\\" + dirs))

        test_images = []
        test_labels = []
        for (dirpath, dirnames, files) in walk(self.setTest):
            test_labels.extend(dirnames)
            break

        for dirs in test_labels:
            test_images.append(self.setTest + "\\" + dirs)

        self.total_validate = len(os.listdir(self.setTest))

        total = self.total_validate + self.total_train

        image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
        validate_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

        train_data_gen = image_generator.flow_from_directory(directory=self.setTrain,
                                                             batch_size=self.batch_size,
                                                             shuffle=True,
                                                             target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                             classes=training_labels,
                                                             class_mode='categorical')
        test_data_gen = validate_generator.flow_from_directory(self.setTest,
                                                               batch_size=self.batch_size,
                                                               shuffle=True,
                                                               target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                               classes=test_labels,
                                                               class_mode='categorical')

        #self.createModel(train_data_gen, test_data_gen)
        self.train(test_data_gen)

    def createModel(self, train_data, test_data):
        """
        Initialize Model
        :return: None
        """
        """
        self.model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(200, 200, 3)),
            MaxPool2D(),
            Dropout(0.2),
            Conv2D(32, 3, padding='same', activation='relu', input_shape=(200, 200, 3)),
            MaxPool2D(),
            Dropout(0.2),
            Flatten(input_shape=(200, 200)),
            Dense(128, activation='relu'),
            Dense(28, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        print("Image Shape: ", train_data.image_shape)
        self.model.build(input_shape=(None, 200, 200, 3))
        """
        self.model = keras.models.load_model('MyModel.h5')
        self.model.summary()

        self.model.fit_generator(train_data,
                                 steps_per_epoch=self.total_train // self.batch_size,
                                 epochs=2,
                                 validation_data=test_data,
                                 validation_steps=self.total_validate // self.batch_size)

        self.model.save("MyModel.h5")

    def evaluate(self, img):
        """
        Evaluate from OpenCV
        and return the sign language letter
        :param img: Frame Image from OpenCV
        :return: sign lanugage letter
        """
        self.model = keras.models.load_model('MyModel.h5')

        self.model.summary()

        predict = self.model.predict(img)

        return predict

    def train(self, test):
        """
        Test data
        :param test:
        :return:
        """
        self.model = keras.models.load_model('MyModel.h5')
        self.model.summary()
        print("Evaluate")
        self.model.evaluate(test)
