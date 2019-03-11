'''
Asynchronous Stitching Script - Parallel version
Author: Gerald M

This script pulls the data generated through TissueCyte (or another microscope system) and
can perfom image averaging correction on the images if requested, before calling ImageJ
from the command line to perform the stitching. You will need to have the plugin script
OverlapY.ijm installed in ImageJ in order for the difference in the X and Y overlap to be
registered. Otherwise the X overlap will be used for both.

The pipeline has been sped up in some areas by parallelising some functions.

Installation:
1) Navigate to the folder containing the AsyncstitchGM.py
2) Run 'pip install -r requirements.txt'

Instructions:
1) Run the script in a Python IDE
2) Fill in the parameters that you are asked for
   Note: You can drag and drop folder paths (works on MacOS) or copy and paste the paths
   Note: The temporary directory is required to speed up ImageJ loading of the files

Updates:
06.03.19 - Updated the overlap and crop parameters to improve the image average result and
tiling artefacts.
11.03.19 - INcluded default values and parameter search from Mosaic file.
'''

import os, sys, warnings, time, glob, errno, subprocess, shutil, math, readline, re
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count, Array, Manager
from functools import partial
from tabCompleter import tabCompleter

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

#=============================================================================================
# Function to load images in parallel
#=============================================================================================
def load_tile(file, cropstart, cropend):
    try:
        tileimage = np.array(Image.open(file).crop((cropstart, cropstart+65, cropend, cropend+65)).rotate(90))
    except (ValueError, IOError, OSError):
        tileimage = np.zeros((cropend-cropstart, cropend-cropstart))
    return tileimage

#=============================================================================================
# Function to check operating system
#=============================================================================================
# This check is to determine which file paths to use if run on the local Mac or Linux supercomputer
def get_platform():
    platforms = {
        'linux1' : 'Linux',
        'linux2' : 'Linux',
        'darwin' : 'Mac',
        'win32' : 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]

if __name__ == '__main__':
    #=============================================================================================
    # Function to check operating system
    #=============================================================================================
    if get_platform() == 'Mac':
        imagejpath = '/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx'
        overlapypath = '"/Applications/Fiji.app/plugins/OverlapY.ijm"'
    if get_platform() == 'Linux':
        imagejpath = '/opt/fiji/Fiji.app/ImageJ-linux64'
        overlapypath = '"/opt/fiji/Fiji.app/plugins/OverlapY.ijm"'
    if get_platform() == 'Windows':
        imagejpath = 'fill in path to imagej executable'
        overlapypath = '"fill in path to OverlapY.ijm"'

    #=============================================================================================
    # Input parameters
    #=============================================================================================
    t = tabCompleter()
    readline.set_completer(t.pathCompleter)
    readline.set_completer_delims('\t')
    readline.parse_and_bind("tab: complete")

    tcpath = raw_input('Select TC data directory (drag-and-drop or tab-complete): ').rstrip()
    temppath = raw_input('Select temporary directory (drag-and-drop or tab-complete): ').rstrip()

    acknowledge = raw_input('Fill in the following variables. To accept default value, leave response blank and press Enter. Press Enter to continue: ')
    startsec = raw_input('Start section (default start): ')
    endsec = raw_input('End section (default end): ')
    xoverlap = raw_input('X overlap % (default 1): ')
    yoverlap = raw_input('Y overlap % (default 1): ')
    channel = raw_input('Channel to stitch: ')
    while not channel:
        channel = raw_input('Channel to stitch: ')
    avgcorr = raw_input('Perform average correction? (y/n): ')
    while avgcorr not in ('y', 'n'):
        avgcorr = raw_input('Perform average correction? (y/n): ')
    convert = raw_input('Perform additional downsize? (y/n): ')
    while convert not in ('y', 'n'):
        convert = raw_input('Perform additional downsize? (y/n): ')
    if convert == 'y':
        downsize = raw_input('Downsize amount (default 0.054 for 10 um/pixel): ')

    # Handle default values
    if not startsec:
        startsec = 1
    else:
         startsec = int(startsec)
    if not xoverlap:
        xoverlap = 1
    else:
        xoverlap = int(xoverlap)
    if not yoverlap:
        yoverlap = 1
    else:
        yoverlap = int(yoverlap)
    if not downsize:
        downsize = 0.054
    else:
        downsize = float(downsize)

    # Search the mosaic file for remaining parameters
    mosaicfile = glob.glob(tcpath+'/Mosaic*.txt')[0]
    with open(mosaicfile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not endsec:
                if 'sections' in line:
                    endsec = int(re.split(':', line.rstrip())[-1])
            if 'Sample ID' in line:
                scanid = re.split(':', line.rstrip())[-1]
            if 'mrows' in line:
                xtiles = int(re.split(':', line.rstrip())[-1])
            if 'mcolumns' in line:
                ytiles = int(re.split(':', line.rstrip())[-1])
            if 'layers' in line:
                zlayers = int(re.split(':', line.rstrip())[-1])

    # Create folders
    os.umask(0000)
    try:
        os.makedirs(tcpath+'/'+str(scanid)+'-Mosaic/Ch'+str(channel)+'_Stitched_Sections', 0777)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if convert == 'y':
        try:
            os.makedirs(tcpath+'/'+str(scanid)+'-Mosaic/Ch'+str(channel)+'_Stitched_Sections_Downsized', 0777)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # Check if temporary folder is empty
    if os.listdir(temppath) != []:
        raise Exception('Temporary folder needs to be empty!')

    crop = 0
    filenamestruct = []
    tstart = time.time()
    zcount = ((startsec-1)*zlayers)+1
    filenumber = 0
    tilenumber = 0
    lasttile = -1
    tileimage = 0

    #=============================================================================================
    # Stitching
    #=============================================================================================
    print ''
    print '------------------------------------------'
    print '          Asynchronous stitching          '
    print '------------------------------------------'

    tstart = time.time()

    # Check that data exists
    for section in range(startsec,endsec+1,1):
        if section <= 9:
            sectiontoken = '000'+str(section)
        elif section <= 99:
            sectiontoken = '00'+str(section)
        else:
            sectiontoken = '0'+str(section)

        folder = scanid+'-'+sectiontoken

        # Token variable hold
        x = xtiles
        y = ytiles
        xstep = -1

        for layer in range(1,zlayers+1,1):
            completelayer = False
            firsttile = xtiles*ytiles*((zlayers*(section-1))+layer-1)
            lasttile = (xtiles*ytiles*((zlayers*(section-1))+layer))-1

            # If last tile doesn't exist yet, wait for it
            if glob.glob(tcpath+'/'+folder+'/*-'+str(lasttile)+'_0*.tif') == []:
                while glob.glob(tcpath+'/'+folder+'/*-'+str(lasttile)+'_0*.tif') == []:
                    sys.stdout.write('\rLast tile not generated yet. Waiting.')
                    sys.stdout.flush()
                    time.sleep(3)
                    sys.stdout.write('\rLast tile not generated yet. Waiting..')
                    sys.stdout.flush()
                    time.sleep(3)
                    sys.stdout.write('\rLast tile not generated yet. Waiting...')
                    sys.stdout.flush()

            # Get file name structure and remove last 8 characters to leave behind filename template
            filenamestruct = glob.glob(tcpath+'/'+folder+'/*-'+str(lasttile)+'_0'+str(channel)+'.tif')[0].rpartition('-')[0]+'-'
            filenames = [filenamestruct+str(tile)+'_0'+str(channel)+'.tif' for tile in range(firsttile, lasttile+1, 1)]

            # Get crop value
            if crop == 0:
                size = Image.open(filenames[-1]).size[0]
                cropstart = int(round(0.04*size))
                cropend = int(round(size-cropstart+1))
            crop = 1

            # Load images through parallel
            pool = Pool(cpu_count())
            img_arr = np.squeeze(np.array([pool.map(partial(load_tile, cropstart=cropstart, cropend=cropend), filenames)]), axis=0).astype(np.float32)
            pool.close()
            pool.join()

            # Compute average tile and divide if required
            if avgcorr == 'y':
                avgimage = (np.mean(img_arr, axis=0)/2)+5
                print 'Computed average tile.'
                #img_arr = np.multiply(np.divide(img_arr, avgimage[np.newaxis, :], out=np.zeros_like(img_arr), where=avgimage[np.newaxis, :]!=0), 100)
                img_arr = np.multiply(np.divide(img_arr, avgimage[np.newaxis, :], out=np.zeros_like(img_arr), where=avgimage[np.newaxis, :]!=0), 100)

            # Save each tile with corresponding name
            for tile_img in img_arr:
                if x>=1 and x<=xtiles:
                    if x<10:
                        xtoken = '00'+str(x)
                    else:
                        xtoken = '0'+str(x)
                    x+=xstep
                    if y<10:
                        ytoken = '00'+str(y)
                    else:
                        ytoken = '0'+str(y)
                elif x>xtiles:
                    x = xtiles
                    if x<10:
                        xtoken = '00'+str(x)
                    else:
                        xtoken = '0'+str(x)
                    xstep*=-1
                    x+=xstep
                    y+=-1
                    if y<10:
                        ytoken = '00'+str(y)
                    else:
                        ytoken = '0'+str(y)
                elif x<1:
                    x=1
                    if x<10:
                        xtoken = '00'+str(x)
                    else:
                        xtoken = '0'+str(x)
                    xstep*=-1
                    x+=xstep
                    y+=-1
                    if y<10:
                        ytoken = '00'+str(y)
                    else:
                        ytoken = '0'+str(y)

                if zcount < 10:
                    ztoken = '00'+str(zcount)
                elif zcount < 100:
                    ztoken = '0'+str(zcount)
                else:
                    ztoken = str(zcount)

                Image.fromarray(tile_img.astype(np.uint16)).save(temppath+'/Tile_Z'+ztoken+'_Y'+ytoken+'_X'+xtoken+'.tif')

            print 'Stitching Z'+ztoken+'...'

            tilepath = temppath+'/'
            stitchpath = tcpath+'/'+scanid+'-Mosaic/Ch'+str(channel)+'_Stitched_Sections'

            outname = "'"+stitchpath+'/Stitched_Z'+ztoken+'.tif'+"'"

            subprocess.call([imagejpath, '--headless', '-eval', 'runMacro('+overlapypath+');', '-eval', 'run("Grid/Collection stitching", "type=[Filename defined position] grid_size_x='+str(xtiles)+' grid_size_y='+str(ytiles)+' tile_overlap_x='+str(xoverlap)+' tile_overlap_y='+str(yoverlap)+' first_file_index_x=1 first_file_index_y=1 directory=['+tilepath+'] file_names=Tile_Z'+ztoken+'_Y{yyy}_X{xxx}.tif output_textfile_name=TileConfiguration_Z'+ztoken+'.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]");', '-eval', 'saveAs("Tiff",'+outname+');'], stdout=open(os.devnull, 'wb'))

            shutil.rmtree(temppath)
            os.makedirs(temppath, 0777)

            if convert == 'y':
                stitched_img = np.array(Image.open(stitchpath+'/Stitched_Z'+ztoken+'.tif'))
                stitched_img = Image.fromarray(np.multiply(np.divide(stitched_img.astype(float),np.max(stitched_img.astype(float))), 255.).astype('uint8'), 'L')
                stitched_img = stitched_img.resize((int(downsize*stitched_img.size[0]), int(downsize*stitched_img.size[1])), Image.ANTIALIAS)
                stitched_img.save(stitchpath+'_Downsized/Stitched_Z'+ztoken+'.tif')

            print 'Complete!'

            zcount+=1
            y = ytiles
            x = xtiles
            xstep = -1

            tilenumber+=1

    #=============================================================================================
    # Finish
    #=============================================================================================

    minutes, seconds = divmod(time.time()-tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    print ''
    print 'Stitching completed in %02d:%02d:%02d:%02d' %(days, hours, minutes, seconds)
