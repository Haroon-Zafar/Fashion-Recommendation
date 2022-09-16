import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
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


# for debugging
# print(model.summary())

def extractFeatures(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    # converting the image to array
    x = image.img_to_array(img)
    # adding one more dimension
    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    # preprocessing the input
    x = preprocess_input(x)
    # passing the input to the model to get the features
    features = model.predict(x, verbose=0)
    return features
