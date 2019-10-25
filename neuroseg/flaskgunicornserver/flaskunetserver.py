# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import model_from_json
# import tensorflow.compat.v1 as tf
import tensorflow as tf
from PIL import Image
from multiprocessing import cpu_count
import numpy as np
import tensorflow.keras.backend as K

# initialize our Flask application and the Keras model
import flask
app = flask.Flask(__name__)

K.clear_session()
K.set_learning_phase(0)

GPU = False

if not GPU:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print ('Using CPU.')
else:
    print ('Using GPU.')

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    global graph

    model_path = '../models/2019_09_30_UNet/focal_unet_model.json'
    weights_path = '../models/2019_09_30_UNet/focal_unet_weights.best.hdf5'

    # Load the classifier model, initialise and compile
    with open(model_path, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)

    model._make_predict_function()
    # graph = tf.get_default_graph()

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
            # with graph.as_default():
            preds = model.predict(image)
            preds = np.squeeze(preds[0])

            # Add prediction to dictionary as a list (array does not work)
            data["predictions"] = preds.tolist()

            # Show the request was successful
            data["success"] = True

    # Return the result
    return flask.jsonify(data)

print("* Loading Keras model and Flask starting server...")

load_model()

if __name__ == "__main__":
    app.run(host='0.0.0.0')
