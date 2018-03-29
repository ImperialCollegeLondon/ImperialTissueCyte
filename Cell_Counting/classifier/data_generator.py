# Image Generator example to test 8- and 16- bit tiff file loading

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(rotation_range=180, rescale = 1./65535, shear_range = 0.2, zoom_range = 0.2, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip = True, vertical_flip = True)

#img = load_img('/Users/gm515/Documents/Python/Machine Learning/cell correction/16-bit/test_data/cell/cell-00015.tif')  # this is a PIL image
img = Image.open('/Users/gm515/Documents/Python/Machine Learning/cell correction/16-bit/test_data/cell/cell-00015.tif')
x = img_to_array(img)*1./65535  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cell', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
