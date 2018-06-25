import os, sys, warnings, time, glob, errno, subprocess, shutil
import numpy as np
from PIL import Image

#=============================================================================================
# Input parameters
#=============================================================================================

tcpath = raw_input('Select TC data directory (drag-and-drop): ').rstrip()
temppath = raw_input('Select temporary directory (drag-and-drop): ').rstrip()

scanid = raw_input('Scan ID: ')
startsec = input('Start section: ')
endsec = input('End section: ')
xtiles = input('Number of X tiles: ')
ytiles = input('Number of Y tiles: ')
zlayers = input('Number of Z layers per slice: ')
xoverlap = input('X overlap % (default 5): ')
yoverlap = input('Y overlap % (default 6): ')
channel = input('Channel to stitch: ')
convert = raw_input('Downsize 0.25? (y/n): ')

# Create folders
try:
    os.makedirs(tcpath+'/'+str(scanid)+'-Mosaic/Ch'+str(channel)+'_Stitched_Sections')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

if convert == 'y':
    try:
        os.makedirs(tcpath+'/'+str(scanid)+'-Mosaic/Ch'+str(channel)+'_Stitched_Sections_0.25')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# Check if temporaty folder is empty
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
        lasttile = xtiles*ytiles*((zlayers*(section-1))+layer)-1

        # If last tile doesn't exist yet, wait for it
        if glob.glob(tcpath+'/'+folder+'/*-'+str(lasttile)+'_0*.tif') == []:
            while glob.glob(tcpath+'/'+folder+'/*-'+str(lasttile)+'_0*.tif') == []:
                sys.stdout.write('\rLast tile not generated yet. Waiting.')
                sys.stdout.flush()
                time.sleep(2)
                sys.stdout.write('\rLast tile not generated yet. Waiting..')
                sys.stdout.flush()
                time.sleep(2)
                sys.stdout.write('\rLast tile not generated yet. Waiting...')
                sys.stdout.flush()

        filenumber = firsttile

        for tile in range(firsttile, lasttile+1, 1):
            # Get file name structure and remove last 8 characters to leave behind filename template
            filenamestruct = glob.glob(tcpath+'/'+folder+'/*-'+str(filenumber)+'_0'+str(channel)+'.tif')[0].rpartition('-')[0]+'-'

            # Try to open file. If it doesn't exist, create an empty file
            try:
                tileimage = Image.open('/'+filenamestruct+str(filenumber)+'_0'+str(channel)+'.tif')
            except IOError as e:
                if e.errno == errno.ENOENT:
                    tileimage = Image.fromarray(np.zeros(tileimage.size))
                else:
                    raise

            # Set crop value if not already stored
            if crop == 0:
                crop = round(0.018*tileimage.size[0])

            # Crop and rotate image and convert to numpy array
            tileimage2 = np.array(tileimage.crop((crop, crop, tileimage.size[0]-crop, tileimage.size[1]-crop)).rotate(-90))

            if tile == firsttile:
                sumimage = tileimage2
            else:
                sumimage = sumimage + tileimage2

            filenumber+=1

        # Compute average tile
        avgimage = (sumimage/(xtiles*ytiles)).astype(float)
        print 'Computed average tile.',

        tilenumber = firsttile

        for tile in range(firsttile, lasttile+1, 1):
            # Try to open file. If it doesn't exist, create an empty file
            try:
                tileimage = Image.open('/'+filenamestruct+str(tilenumber)+'_0'+str(channel)+'.tif')
            except IOError as e:
                if e.errno == errno.ENOENT:
                    tileimage = Image.fromarray(np.zeros(tileimage.size))
                else:
                    raise

            tileimage2 = np.array(tileimage.crop((crop, crop, tileimage.size[0]-crop, tileimage.size[1]-crop)).rotate(-90))
            tileimage2 = Image.fromarray((tileimage2).astype(np.uint16))

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

            tileimage2.save(temppath+'/Tile_Z'+ztoken+'_Y'+ytoken+'_X'+xtoken+'.tif')

            if (tile+1)%(xtiles*ytiles) == 0:
                print 'Stitching Z'+ztoken+'...',

                tilepath = temppath+'/'
                stitchpath = tcpath+'/'+scanid+'-Mosaic/Ch'+str(channel)+'_Stitched_Sections'
                subprocess.call(['/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx', '--headless', '-eval', 'run("Grid/Collection stitching", "type=[Filename defined position] grid_size_x='+str(xtiles)+' grid_size_y='+str(ytiles)+' tile_overlap_x='+str(xoverlap)+' tile_overlap_y='+str(yoverlap)+' first_file_index_x=1 first_file_index_y=1 directory='+tilepath+' file_names=Tile_Z'+ztoken+'_Y{yyy}_X{xxx}.tif output_textfile_name=TileConfiguration_Z'+ztoken+'.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory='+stitchpath+'") '], stdout=open(os.devnull, 'wb'))


                os.rename(stitchpath+'/img_t1_z1_c1', stitchpath+'/Stitched_Z'+ztoken+'.tif')

                if convert == 'y':
                    stitched_img = Image.open(stitchpath+'/Stitched_Z'+ztoken+'.tif')
                    stitched_img.resize(0.25*stitched_img.size)
                    stitched_img.save(stitchpath+'0.25/Stitched_Z'+ztoken+'.tif')

                print 'Complete!'

                shutil.rmtree(temppath)
                os.makedirs(temppath)

                zcount+=1
                y = ytiles
                x = xtiles
                xstep+=-1

            tilenumber+=1

#=============================================================================================
# Finish
#=============================================================================================

minutes, seconds = divmod(time.time()-tstart, 60)
hours, minutes = divmod(minutes, 60)
days, hours = divmod(hours, 24)

print ''
print 'Stitching completed in %02d:%02d:%02d:%02d' %(days, hours, minutes, seconds)
