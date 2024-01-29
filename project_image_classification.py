import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test, y_train, y_test):
    X_train = X_train.reshape((X_train.shape[0], -1)).astype('float32') / 255
    X_test = X_test.reshape((X_test.shape[0], -1)).astype('float32') / 255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test

def create_baseline_model():
    model = Sequential()
    model.add(Dense(256, input_dim=784, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def compile_and_train_model(model, X_train, y_train):
    opt = SGD(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
    return model

def evaluate_model(model, X_test, y_test):
    scores = model.evaluate(X_test, y_test, verbose=1)
    print(f"Error: {100 - scores[1] * 100:.2f}%")

def load_and_predict_image(model, image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    img_array = img_to_array(img)
    img_array = img_array.reshape((1, -1)).astype('float32') / 255
    preds = model.predict(img_array)
    prob = model.predict_proba(img_array)
    print('Predicted value is:', np.argmax(preds[0]))

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist_data()
    X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)

    model = create_baseline_model()
    model.summary()

    model = compile_and_train_model(model, X_train, y_train)

    evaluate_model(model, X_test, y_test)

    # Replace the image path with the correct path to your image
    image_path = "path/to/your/image.png"
    load_and_predict_image(model, image_path)
