import SimpleITK as sitk
import time
import argparse
import numpy as np
from skimage import io

def slack_message(text, channel, username):
    from urllib import request, parse
    import json

    post = {"text": "{0}".format(text),
        "channel": "{0}".format(channel),
        "username": "{0}".format(username),
        "icon_url": "https://github.com/gm515/gm515.github.io/blob/master/Images/imperialstplogo.png?raw=true"}

    try:
        json_data = json.dumps(post)
        req = request.Request('https://hooks.slack.com/services/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',
            data=json_data.encode('ascii'),
            headers={'Content-Type': 'application/json'})
        resp = request.urlopen(req)
    except Exception as em:
        print("EXCEPTION: " + str(em))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('autoflpath', default=[], type=str, help='File path for autofluorescence atlas')
    parser.add_argument('-avgpath', default='atlases/average_10um.tif', type=str, help='File path for average atlas')
    parser.add_argument('-annopath', default='atlases/annotation_10um.tif', type=str, help='File path for annotation atlas')
    parser.add_argument('-hempath', default='atlases/hemisphere_10um.tif', type=str, help='File path for hemisphere atlas')
    parser.add_argument('-first', default=1, type=int, dest='first', help='First slice in average atlas')
    parser.add_argument('-last', default=1320, type=int, dest='last', help='Last slice in average atlas')
    parser.add_argument('-fixedpts', default=None, type=str, dest='fixedpointspath', help='Path for fixed points')
    parser.add_argument('-movingpts', default=None, type=str, dest='movingpointspath', help='path for moving points')


    args = parser.parse_args()

try:
    print ('Loading all atlases...')
    fixedData = sitk.ReadImage(args.autoflpath)
    print ('Autofluorescence atlas loaded')
    movingData = sitk.GetImageFromArray(io.imread(args.avgpath)[args.first:args.last+1,:,:])
    print ('Average atlas loaded')
    annoData = sitk.GetImageFromArray(io.imread(args.annopath)[args.first:args.last+1,:,:])
    print ('Annotation atlas loaded')
    hemData = np.zeros(movingData.GetSize())
    hemData[570::,:,:] = 1
    hemData = sitk.GetImageFromArray(np.uint8(np.swapaxes(hemData,0,2)))
    print ('Hemisphere atlas loaded')

    tstart = time.time()

    # Initiate SimpleElastix
    SimpleElastix = sitk.ElastixImageFilter()
    SimpleElastix.LogToFileOn()
    SimpleElastix.SetOutputDirectory('output/')
    SimpleElastix.SetFixedImage(fixedData)
    SimpleElastix.SetMovingImage(movingData)

    # SimpleElastix.SetFixedPointSetFileName(args.fixedpointspath)
    # SimpleElastix.SetMovingPointSetFileName(args.movingpointspath)

    # Create segmentation map
    parameterMapVector = sitk.VectorOfParameterMap()

    # Start with Affine using fixed points to aid registration
    affineParameterMap = sitk.ReadParameterFile('02_ARA_affine.txt')
    # affineParameterMap['Metric'] = ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"] # uncomment to ignore the corresponding points
    parameterMapVector.append(affineParameterMap)

    # Add very gross BSpline to make rough adjustments to the affine result
    bsplineParameterMap = sitk.ReadParameterFile('par0025bspline.modified.txt')
    # bsplineParameterMap['Metric'] = ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"] # uncomment to ignore the corresponding points
    parameterMapVector.append(bsplineParameterMap)

    # bsplineParameterMap = sitk.ReadParameterFile('par0025bspline.modified.txt')
    # bsplineParameterMap["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    # bsplineParameterMap['Metric'] = ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"]
    # bsplineParameterMap['FinalGridSpacingInVoxels'] = ['50.0', '50.0', '50.0']
    # bsplineParameterMap['MaximumNumberOfIterations'] = ['200']
    # parameterMapVector.append(bsplineParameterMap)

    # Set the parameter map
    SimpleElastix.SetParameterMap(parameterMapVector)

    # Print parameter map
    SimpleElastix.PrintParameterMap()

    # Execute
    SimpleElastix.Execute()

    # Save average transform
    averageSeg = SimpleElastix.GetResultImage()

    # Get transform map and apply to segmentation data
    transformMap = SimpleElastix.GetTransformParameterMap()

    resultSeg = sitk.Transformix(annoData, transformMap)

    hemSeg = sitk.Transformix(hemData, transformMap)

    # Write average transform and segmented results
    # sitk.WriteImage(averageSeg, args.autoflpath+'AVGRES.tif')
    sitk.WriteImage(resultSeg, args.autoflpath+'SEGRES.tif')
    # sitk.WriteImage(hemSeg, args.autoflpath+'HEMRES.tif')

    minutes, seconds = divmod(time.time()-tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    text = args.autoflpath+'\n SimpleElastix segmentation completed in %02d:%02d:%02d:%02d' %(days, hours, minutes, seconds)

    print ('')
    print (text)

    slack_message(text, '#segmentation', 'Segmentation')

except (RuntimeError, TypeError, NameError, ImportError, SyntaxError, FileNotFoundError):
    slack_message('*ERROR*', '#segmentation', 'Segmentation')
