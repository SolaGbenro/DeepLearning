"""
    Sola Gbenro:
        Deep Learning A-Z -- Part 2 Convolutional Neural Networks
"""
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# # each of the training and testing directories has sub directories 'cats' and 'dogs'
PATH_TO_TEST = os.getcwd() + "\dataset\\test_set"
PATH_TO_TRAIN = os.getcwd() + "\dataset\\training_set"
PATH_TO_MODEL = os.getcwd() + "\models"

# apply geometric transformations to training set images (flip, zoom, shear etc)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# apply Image Data Generator object (train_datagen) to training_set
training_set = train_datagen.flow_from_directory(
    PATH_TO_TRAIN,
    # reduce pixel size to reduce training time
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

# rescale testing images
test_datagen = ImageDataGenerator(rescale=1./255)
# apply Image Data Generator object (test_datagen) to testing_set
test_set = test_datagen.flow_from_directory(
    PATH_TO_TEST,
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

# initialize the model
cnn = tf.keras.models.Sequential()
# add Convolutional layer, 32 filters, 3x3 kernel size, images have been resized to 64x64 RGB
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
# add pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
# add 2nd Conv layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
# add 2nd pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
# flatten into 1d vector
cnn.add(tf.keras.layers.Flatten())
# add fully connected layer, 128 hidden neurons
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# add the final output layer, Binary classification requires 1 neuron and sigmoid
# multi-class classification requires (# of classes - 1) neurons and softmax
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# compile the model (connect to optimizer a loss function and additional metrics)
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# save model
cnn.save(PATH_TO_MODEL)
"""
    Load a saved Model
from tensorflow import keras
model = keras.models.load_model(PATH_TO_MODEL)
"""

""" Make a single Prediction """
# resize single testing image to match 64,64 dimensions from above
single_test_image_1 = image.load_img(os.getcwd() + "\dataset\single_prediction\cat_or_dog_1.jpg", target_size=(64, 64))
single_test_image_2 = image.load_img(os.getcwd() + "\dataset\single_prediction\cat_or_dog_2.jpg", target_size=(64, 64))
# convert test image from PIL to numpy format
single_test_image_1 = image.img_to_array(single_test_image_1)
single_test_image_2 = image.img_to_array(single_test_image_2)

# predict method for our model requires batches (even if only 1 test image in batch)
# convert single image into a batch, adding an additional dimension to the first axis (dimension 1 is expanded)
single_test_image_1 = np.expand_dims(single_test_image_1, axis=0)
single_test_image_2 = np.expand_dims(single_test_image_2, axis=0)
# results
result_1 = cnn.predict(single_test_image_1)
result_2 = cnn.predict(single_test_image_2)

# 0 == cat, 1 == dog
training_set.class_indices
# first element inside first batch (only element and only batch)
if result_1[0][0] == 1:
    prediction_1 = 'dog'
else:
    prediction_1 = 'cat'
# second test_image
if result_2[0][0] == 1:
    prediction_2 = 'dog'
else:
    prediction_2 = 'cat'

# view prediction of single test images
print(f"First prediction is {prediction_1}")
print(f"Second prediction is {prediction_2}")
