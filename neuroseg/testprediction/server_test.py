import numpy as np
import math
import requests
import time
from PIL import Image

KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = '00007.tif'

# image = open(IMAGE_PATH, 'rb').read()

images_array = []

image = np.array(Image.open(IMAGE_PATH)).astype(np.float32)
image = (image-np.min(image))/(np.max(image)-np.min(image))

# Image.fromarray(np.uint8(image*255)).save('/Users/gm515/Desktop/img/'+str(slice_number)+'.tif')

shape = image.shape
newshape = tuple((int( 16 * math.ceil( i / 16. )) for i in shape))
image = np.pad(image, ((0,np.subtract(newshape,shape)[0]),(0,np.subtract(newshape,shape)[1])), 'constant')

images_array.append(image)
images_array = np.array(images_array)
images_array = images_array[..., np.newaxis]

image_payload = bytes(images_array)

shape_payload = bytes(np.array(images_array.shape))

# Make payload for request
payload = {"image": image_payload, "shape": shape_payload}

# Submit the request
tstart = time.time()
r = requests.post(KERAS_REST_API_URL, files=payload).json()
print (time.time()-tstart)
# ensure the request was sucessful
if r["success"]:
    print("Request successful")
    # loop over the predictions and display them
    mask = r["predictions"]

    # print(mask)

# otherwise, the request failed
else:
    print("Request failed")
