import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Creating the ResnEt50 model and loading the weights, top layer is false because we will add own ours
# Standard Image size 224, 224, 3

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))

# our model is already trained on imagenet, so we will freeze the layers
# using model to predict the image
# only adding the top layers

model.trainable = False

# passing our model , and adding the top layers


# for debugging
# print(model.summary())

def extractFeatures(img_path):

    # we have to preprocess the image before passing it to the model
    # we will use the preprocess_input function from keras
    img = image.load_img(img_path, target_size=(224, 224))

    # converting the image to array
    imgArray = image.img_to_array(img)

    # converting imgArray dimensions 224,224,3 to 1,224,224,3
    # why ? because keras work on batches of images not single image
    # if you have 100 images then you will have 100,224,224,3
    # if you have single image you have to tell keras that it is a batch of 1 image
    # resizing images
    # will give 4D array
    expandedImgArray = np.expand_dims(imgArray, axis=0)

    # preprocessing the image
    # preprocess_input is a function that will format the image into the format that the model expects
    preProcessedImg = preprocess_input(expandedImgArray)

    # predict() function enables us to predict the labels of the data values on the basis of the trained model.
    result = model.predict(preProcessedImg, verbose=0).flatten()

    # normalized result
    normalizedResult = result / norm(result)

    return (normalizedResult)


# Now we are making a list in which we place the file names of the images in the folder `image`
# I want to print the names of the images in the folder `image` in the terminal

fileNames = []

# print(os.listdir('images'))

# for loop will traverse through every file in the folder `images`
# we are appending file names to the list `fileNames`

for file in os.listdir('images'):
    fileNames.append(os.path.join('images', file))

# print(len(fileNames))
# print(fileNames[0:5])


# we just have to call this extractFeatures function for every image in the folder `images`
# it will return a list of features for every image in the folder `images`

# we will store the features in a list
featuresList = []

# tqdm is a progress bar library in python, here it tells progress of for loop

for file in tqdm(fileNames):
    featuresList.append(extractFeatures(file))


print(np.array(featuresList).shape)


# Dumping the featuresList into a file
# wb = write binary

pickle.dump(featuresList, open('featuresList.pkl', 'wb'))


# Dumping the filenames into a file

pickle.dump(fileNames, open('featuresList.pkl', 'wb'))
