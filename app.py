import tensorflow as tf
from tensorflow.keras.preprocessing.image import image
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Creating the ResnEt50 model and loading the weights, top layer is false because we will add own ours
# Standard Image size 224, 224, 3

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))

# our model is already trained on imagenet, so we will freeze the layers
# using model to predict the image
# only adding the top layers

model.trainable = False

# passing our model , and adding the top layers

model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

print(model.summary())
