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

    autoflpath = '/mnt/TissueCyte80TB/181024_Gerald_HET/het-Mosaic/het181024_autofl_10um.tif'
    avgpath = 'atlases/average_10um.tif'
    annopath = 'atlases/annotation_10um.tif'
    first = 1
    last = 1320

    bsplineparams = ['par0025bspline.modified.txt']#, 'par0033bspline.txt', 'par0038bspline.txt', 'par0025bspline.txt']

    for paramap in bsplineparams:
        try:
            print ('Loading all atlases...')
            fixedData = sitk.GetImageFromArray(255-io.imread(autoflpath))
            print ('Autofluorescence atlas loaded')
            movingData = sitk.GetImageFromArray(65535-io.imread(avgpath)[first:last+1,:,:])
            print ('Average atlas loaded')
            annoData = sitk.GetImageFromArray(io.imread(annopath)[first:last+1,:,:])
            print ('Annotation atlas loaded')

            tstart = time.time()

            # Initiate SimpleElastix
            SimpleElastix = sitk.ElastixImageFilter()
            SimpleElastix.LogToFileOn()
            SimpleElastix.SetOutputDirectory('output/')
            SimpleElastix.SetFixedImage(fixedData)
            SimpleElastix.SetMovingImage(movingData)

            # Create segmentation map
            parameterMapVector = sitk.VectorOfParameterMap()

            # Start with Affine using fixed points to aid registration
            affineParameterMap = sitk.ReadParameterFile('02_ARA_affine.txt')
            parameterMapVector.append(affineParameterMap)

            # Add very gross BSpline to make rough adjustments to the affine result
            bsplineParameterMap = sitk.ReadParameterFile(paramap)
            parameterMapVector.append(bsplineParameterMap)

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

            # Write average transform and segmented results
            # sitk.WriteImage(averageSeg, args.autoflpath+'AVGRES.tif')
            sitk.WriteImage(resultSeg, autoflpath+'SEGRES'+paramap+'.tif')

            minutes, seconds = divmod(time.time()-tstart, 60)
            hours, minutes = divmod(minutes, 60)
            days, hours = divmod(hours, 24)
            text = paramap+' '+autoflpath+'\n SimpleElastix segmentation completed in %02d:%02d:%02d:%02d' %(days, hours, minutes, seconds)

            print ('')
            print (text)

            slack_message(text, '#segmentation', 'Segmentation')

        except (RuntimeError, TypeError, NameError, ImportError, SyntaxError, FileNotFoundError):
            slack_message('*ERROR*', '#segmentation', 'Segmentation')
