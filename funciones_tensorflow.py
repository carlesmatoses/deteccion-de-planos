import os
import numpy as np
# from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image as image_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19

from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


import tensorflow as tf 
import tensorflow_hub as hub
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")



def model_predict(args,MODELS,folder_path,pred_num=1):
    
    # initialize the input image shape (224x224 pixels) along with
    # the pre-processing function (this might need to be changed
    # based on which model we use to classify our image)
    inputShape = (224, 224)
    preprocess = imagenet_utils.preprocess_input
    # if we are using the InceptionV3 or Xception networks, then we
    # need to set the input shape to (299x299) [rather than (224x224)]
    # and use a different image pre-processing function
    if args.model in ("inception", "xception"):
        inputShape = (299, 299)
        preprocess = preprocess_input

    # load our the network weights from disk (NOTE: if this is the
    # first time you are running this script for a given network, the
    # weights will need to be downloaded first -- depending on which
    # network you are using, the weights can be 90-575MB, so be
    # patient; the weights will be cached and subsequent runs of this
    # script will be *much* faster)
    print("[INFO] loading {}...".format(args.model))
    Network = MODELS[args.model]
    model = Network(weights="imagenet")

    # bucle para escenas
    path = os.listdir(f"{args.output}/high_res")
    predictions = []
    for img_path in path:
        img = os.path.join(f"{args.output}/high_res",img_path)
        # load the input image using the Keras helper utility while ensuring
        # the image is resized to `inputShape`, the required input dimensions
        # for the ImageNet pre-trained network
        print("[INFO] loading and pre-processing image...")
        image = load_img(img, target_size=inputShape)
        image = img_to_array(image)

        # our input image is now represented as a NumPy array of shape
        # (inputShape[0], inputShape[1], 3) however we need to expand the
        # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
        # so we can pass it through the network
        image = np.expand_dims(image, axis=0)

        # pre-process the image using the appropriate function based on the
        # model that has been loaded (i.e., mean subtraction, scaling, etc.)
        image = preprocess(image)

        # classify the image
        print("[INFO] classifying image with '{}'...".format(args.model))
        preds = model.predict(image)
        P = imagenet_utils.decode_predictions(preds,pred_num)
        predictions.append(P[0])
    return predictions

def inception_predict(folder_path,pred_num=1):
    """
    folder path
    """
    prediction = []
    path = os.listdir(folder_path)
    model = InceptionV3(weights="imagenet")
    for img_path in path:
        img = image.load_img(os.path.join(folder_path,img_path), target_size=(299, 299))

        #convert image to array
        input_img = image.img_to_array(img)
        input_img = np.expand_dims(input_img, axis=0)
        input_img = inception_v3.preprocess_input(input_img)

        #convert image to array
        input_img = image.img_to_array(img)
        input_img = np.expand_dims(input_img, axis=0)
        input_img = inception_v3.preprocess_input(input_img)


        #Predict the inputs on the model
        predict_img = model.predict(input_img)
        #Let's predict top 5 results
        top = inception_v3.decode_predictions(predict_img, top=pred_num)
        prediction.append(top[0])

    return prediction