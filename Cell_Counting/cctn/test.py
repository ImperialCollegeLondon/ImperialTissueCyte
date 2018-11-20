from PIL import Image
import numpy as np
from filters.gaussmedfilt import gaussmedfilt
from filters.circthresh import circthresh
import scipy.ndimage as ndimage

# Gets image in same format as CCTN
image = Image.open('/Users/gm515/Desktop/test/Stitched_Z559_Cropped.tif')
image = np.array(image).astype(float)
image = np.multiply(np.divide(image,np.max(image)), 255.)
Image.fromarray(np.uint8(image)).save('/Users/gm515/Desktop/test/Stitched_Z559_Cropped_8bit.tif')

# Apply gaussian filter
gauss_image = ndimage.gaussian_filter(image, sigma=(3, 3))
Image.fromarray(np.uint8(gauss_image)).save('/Users/gm515/Desktop/test/Stitched_Z559_Cropped_8bit_Gauss.tif')

# Circularity
image = np.multiply(np.divide(gauss_image, np.max(image)), 255.)
Image.fromarray(np.uint8(gauss_image)).save('/Users/gm515/Desktop/test/Stitched_Z559_Cropped_8bit_Gauss_8bit.tif')
circ_image = image>circthresh(image,200,0)
Image.fromarray(np.uint8(circ_image)).save('/Users/gm515/Desktop/test/Stitched_Z559_Cropped_8bit_Gauss_8bit_Circ.tif')
