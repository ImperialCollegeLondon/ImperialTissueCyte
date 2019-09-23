import SimpleITK as sitk
import time
import argparse
import numpy as np
from skimage import io

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-fixed', type=str, help='Fixed data')
    parser.add_argument('-moving', type=str, help='Moving data')

    args = parser.parse_args()

    print ('Loading all atlases...')
    fixedData = sitk.ReadImage(args.fixed)
    print ('Fixed data loaded')
    movingData = sitk.ReadImage(args.moving)
    print ('Moving data loaded')

    tstart = time.time()

    # Initiate SimpleElastix
    SimpleElastix = sitk.ElastixImageFilter()
    # SimpleElastix.LogToFileOn()
    # SimpleElastix.SetOutputDirectory('output/')
    SimpleElastix.SetFixedImage(fixedData)
    SimpleElastix.SetMovingImage(movingData)

    # Create segmentation map
    parameterMapVector = sitk.VectorOfParameterMap()

    # Add very gross BSpline to make rough adjustments to the affine result
    bsplineParameterMap = sitk.ReadParameterFile('par0025bspline.modified.txt')
    bsplineParameterMap['Transform'] = ['AffineTransform']
    bsplineParameterMap['NumberOfResolutions'] = ['0']
    bsplineParameterMap['MaximumNumberOfIterations'] = ['0']
    bsplineParameterMap['FixedImageDimension'] = ['1']
    bsplineParameterMap['MovingImageDimension'] = ['1']
    parameterMapVector.append(bsplineParameterMap)

    # Set the parameter map
    SimpleElastix.SetParameterMap(parameterMapVector)

    # Print parameter map
    SimpleElastix.PrintParameterMap()

    # Execute
    SimpleElastix.Execute()

    minutes, seconds = divmod(time.time()-tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    text = '\n SimpleElastix measure completed in %02d:%02d:%02d:%02d' %(days, hours, minutes, seconds)

    print ('')
    print (text)
