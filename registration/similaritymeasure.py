import argparse
import numpy as np
from scipy import signal
from PIL import Image
from skimage.measure import compare_ssim
import pandas as pd
from skimage import io

def ssim(image1, image2):
    return compare_ssim(image1.reshape(-1), image2.reshape(-1))

def mse(image1, image2):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((image1.reshape(-1) - image2.reshape(-1)) ** 2)
	err /= float(image1.reshape(-1).shape[0])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def cc(image1, image2):
    return np.corrcoef(image1.reshape(-1), image2.reshape(-1))[0,1]

def mi(image1, image2):
    hgram, x_edges, y_edges = np.histogram2d(image1.reshape(-1), image2.reshape(-1), bins=20)
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-fixed', type=str, help='Fixed data')
    parser.add_argument('-moving', type=str, help='Moving data')

    args = parser.parse_args()

    fixedImage = io.imread(args.fixed).astype(np.float32)
    fixedImage /= np.max(fixedImage)

    movingImage = io.imread(args.moving).astype(np.float32)
    movingImage /= np.max(movingImage)

    df = pd.DataFrame()
    df['Metric'] = ['Structural Similarity', 'Mean Square Error', 'Cross-correlation', 'Mutual Information']

    df['Value'] = [ssim(fixedImage[::100,:,:], movingImage[::100,:,:]), mse(fixedImage, movingImage), cc(fixedImage, movingImage), mi(fixedImage, movingImage)]

    print ('Note: SSIM samples every 10 slices to save computation')
    print (df)
