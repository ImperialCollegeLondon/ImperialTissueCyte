# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import model_from_json
import tensorflow as tf
from PIL import Image
import numpy as np
import flask
import io
import requests
import keras.backend as K

K.set_learning_phase(0)
K.set_session(tf.Session())

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    global graph

    model_path = 'models/2019_09_30_UNet/focal_unet_model.json'
    weights_path = 'models/2019_09_30_UNet/focal_unet_weights.best.hdf5'

    # Load the classifier model, initialise and compile
    with open(model_path, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)

    model._make_predict_function()
    graph = tf.get_default_graph()

@app.route("/predict", methods=["POST"])
def predict():
    # Initialise return dictionary
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # Read image in PIL format
            image = flask.request.files["image"].read()
            shape = flask.request.files["shape"].read()
            image = np.frombuffer(image, np.float32)
            shape = tuple(np.frombuffer(shape, np.int))

            image = image.reshape(shape)

            # Predict
            with graph.as_default():
                preds = model.predict(image)
            preds = np.squeeze(preds[0])

            # Add prediction to dictionary as a list (array does not work)
            data["predictions"] = preds.tolist()

            # Show the request was successful
            data["success"] = True

    # Return the result
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
# This should go into the main function of cctn_unet
if __name__ == "__main__":
    print("* Loading Keras model and Flask starting server...")

    load_model()

    app.run(threaded=True)
