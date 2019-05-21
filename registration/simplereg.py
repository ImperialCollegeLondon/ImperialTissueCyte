import SimpleITK as sitk
import time
from skimage import io

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('autoflpath', default=[], type=str, help='File path for autofluorescence atlas')
    parser.add_argument('avgpath', default=[], type=str, help='File path for average atlas')
    parser.add_argument('annopath', default=[], type=str, help='File path for annotation atlas')
    parser.add_argument('hempath', default=[], type=str, help='File path for hemisphere atlas')
    parser.add_argument('first', default=[], type=int, help='First slice in average atlas')
    parser.add_argument('last', default=[], type=int, help='Last slice in average atlas')
    parser.add_argument('outpath', default=[], type=str, help='Directory path to save output files')

    args = parser.parse_args()

    print ('Loading all atlases...')
    fixedData = sitk.ReadImage(args.autoflpath)
    print ('Autofluorescence atlas loaded')
    movingData = io.imread(mask_path)

    end
    sitk.ReadImage(args.avgpath)
    print ('Average atlas loaded')
    annoData = sitk.ReadImage(args.annopath)
    print ('Annotation atlas loaded')
    hemData = sitk.ReadImage(args.hempath)
    print ('Hemisphere atlas loaded')

    tstart = time.time()
