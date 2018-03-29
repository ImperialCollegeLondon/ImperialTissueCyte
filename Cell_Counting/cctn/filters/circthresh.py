import numpy
import cv2

def circthresh(A,SIZE):
    # Convert image to 8-bit
    A = A*255
    A = numpy.uint8(A)

    # Define parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Create detector w/ parameters using ver=3 syntax
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs in A
    keypoints = detector.detect(A)


    return keypoints
