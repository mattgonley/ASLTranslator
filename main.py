import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd


def load_model():
    m = keras.Sequential()
    # add the hidden layers to the CNN
    m.add(layers.Conv2D(64, 3, activation='relu',
                        input_shape=(200, 200, 1)))  # hidden layers
    m.add(layers.MaxPooling2D((2, 2)))
    m.add(layers.Conv2D(64, 1, activation='relu'))  # more layers ...
    m.add(layers.MaxPooling2D((2, 2)))
    m.add(layers.Conv2D(64, 1, activation='relu'))
    m.add(layers.Dropout(0.1))
    m.add(layers.Flatten())
    m.add(layers.Dense(64, activation='relu'))
    m.add(layers.Dense(32, activation='relu'))
    m.add(layers.Dense(3, activation='softmax'))  # get the softmax

    m.load_weights('./notebooks/cnn.h5')
    return m


if __name__ == '__main__':
    model = load_model()
    labels = ['A', 'B', 'C']
    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()
        frame = cv2.resize(frame, (200, 200))
        if not ret:
            print('Video Stream Ended')
            break

        color = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = gray / 255.0
        gray = gray.reshape((1, 200, 200, -1))
        pred = model.predict(gray)
        probs = model.predict_proba(gray)
        pred = np.argmax(pred, axis=1)

        print('Prediction:', labels[pred[0]])

        cv2.imshow('frame', color)

        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
