import tensorflow as tf
from tensorflow.keras.preprocessing.image import image
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
