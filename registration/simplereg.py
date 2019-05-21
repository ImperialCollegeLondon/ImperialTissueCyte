import SimpleITK as sitk
import time
import argparse
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
    parser.add_argument('outpath', default=[], type=str, help='Directory path to save output files')
    parser.add_argument('-avgpath', default='atlases/average_10um.tif', type=str, help='File path for average atlas')
    parser.add_argument('-annopath', default='atlases/annotation_10um.tif', type=str, help='File path for annotation atlas')
    parser.add_argument('-hempath', default='atlases/hemisphere_10um.tif', type=str, help='File path for hemisphere atlas')
    parser.add_argument('-first', default=1, type=int, dest='first', help='First slice in average atlas')
    parser.add_argument('-last', default=1140, type=int, dest='last', help='Last slice in average atlas')

    args = parser.parse_args()

try:

    print ('Loading all atlases...')
    fixedData = sitk.ReadImage(args.autoflpath)
    print ('Autofluorescence atlas loaded')
    movingData = sitk.GetImageFromArray(io.imread(args.avgpath)[:,:,first:last+1])
    print ('Average atlas loaded')
    annoData = sitk.GetImageFromArray(io.imread(args.annopath)[:,:,first:last+1])
    print ('Annotation atlas loaded')
    hemData = sitk.GetImageFromArray(io.imread(args.hempath)[:,:,first:last+1])
    print ('Hemisphere atlas loaded')

    tstart = time.time()

    # Initiate SimpleElastix
    SimpleElastix = sitk.ElastixImageFilter()
    SimpleElastix.LogToFileOn()
    SimpleElastix.SetFixedImage(fixedData)
    SimpleElastix.SetMovingImage(movingData)

    # Create segmentation map
    parameterMapVector = sitk.VectorOfParameterMap()

    # Start with Affine using fixed points to aid registration
    affineParameterMap = sitk.GetDefaultParameterMap('affine')
    parameterMapVector.append(affineParameterMap)

    # Add very gross BSpline to make rough adjustments to the affine result
    bsplineParameterMap = sitk.ReadParameterFile('ARA_bspline_params_10um.txt')
    parameterMapVector.append(bsplineParameterMap)

    # Set the parameter map
    SimpleElastix.SetParameterMap(parameterMapVector)

    # Print parameter map
    SimpleElastix.PrintParameterMap()

    # Execute
    SimpleElastix.Execute()

    # Get transform map and apply to segmentation data
    transformMap = SimpleElastix.GetTransformParameterMap()

    resultSeg = sitk.Transformix(annoData, transformMap)

    hemSeg = sitk.Transformix(hemData, transformMap)

    # Write average transform and segmented results
    sitk.WriteImage(resultSeg, args.outpath+'/segres_10um.tif')
    sitk.WriteImage(hemSeg, args.outpath+'/hemres_10um.tif')

except (RuntimeError, TypeError, NameError, ImportError, SyntaxError):
    slack_message('SimpleElastix segmentation ERROR', '#segmentation', 'Segmentation')

minutes, seconds = divmod(time.time()-tstart, 60)
hours, minutes = divmod(minutes, 60)
days, hours = divmod(hours, 24)
text = 'SimpleElastix segmentation completed in %02d:%02d:%02d:%02d' %(days, hours, minutes, seconds)

print ('')
print (text)

slack_message(text, '#segmentation', 'Segmentation')
