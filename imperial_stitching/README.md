# About Imperial TissueVision Stitching Files
This wiki contains details about the stitching scripts written at Imperial College London to improve upon the original stitching pipeline (see [Original TissueVision Stitching Files](https://github.com/ImperialCollegeLondon/ImperialTissueCyte/tree/master/Original_TissueVision_Stitching_Files)).

Stitching scripts are written in both MATLAB (deprecated) and Python version 3, and both require the Grid/Collection stitching plugin in ImageJ/Fiji. To enable MATLAB to communicate with ImageJ/Fiji, a Java package called MIJ (MATLAB-ImageJ) needs to be installed (if using MATLAB) (see below for installation instructions) and MATLAB also needs to run with a Java version 8 environment. Due to these requirements, there is a sequence of set-up steps that need to be performed before the MATLAB stitching can be executed. **The Python scripts require much less set-up and is recommended. The only requirement is to install the required list of packages.**

Unfortunately, there exists a bug with ImageJ/Fiji Bioformats (an image reading module) which struggles to import images which reside on a network drive. The code therefore has a loophole which saves the pre-processed image data to a local directory. However, this requires at least 1 GB of available space on your local hard drive to avoid storage issues.

Image stitching is a computationally intensive task, made more so due to the size of the image tiles acquired with TissueCyte. It is recommended to have a workstation with at least 16 GB of RAM. Currently, the scripts have been confirmed to run on *macOS* and *Linux*, but many of the instructions below should have corresponding instructions in a *Windows* operating system environment.

The Grid/Collection stitching plugin has a _secret_ parameter which can be added allowing the overlap in X and Y to be separately defined. This is enabled by using the `OverlapY.ijm` file which is a bean shell file that is run in ImageJ/Fiji to enable that option. This should be automatically run during the stitching process so requires no additional steps, other than the file being present in the directory upon running the stitching pipeline.

To set up your workstation to perform the stitching, follow the required installation steps for both MATLAB (deprecated) and Python below, then follow the more specific instructions for your programming language of choice.

# Installation required for both MATLAB (deprecated) and Python 3 Versions
The following is required for both the MATLAB (deprecated) and Python version of the stitching pipeline.

## Fiji set-up
1. Download ImageJ/Fiji if not already installed (get Fiji which is a glorified ImageJ, not just the basic ImageJ version).
2. In Fiji, check the “Grid/Collection stitching” plugin is installed under `Plugins > Grid/Collection stitching`.
3. Next, the available RAM for Fiji needs to be increased to avoid memory shortage errors during the stitching. Go to `Edit > Options > Memory & Threads…`. The window which opens should contain a “Maximum Memory” field which can edited to a specific value. Change this to no more than 3/4 of your available system RAM. For example, a 32 GB workstation can be set to have 24000 MB of memory available for Fiji. Avoid exceeding the 3/4 rule as this can make your workstation unstable due to other background processes which utilise RAM.
4. Close Fiji for the effect to take place.
5. Finally, download the `OverlapY.ijm` file and move it to the `Plugins` folder of your ImageJ/Fiji application. On MacOS this is found by right clicking the application and showing package contents. For Windows, you might need to locate this path another way. Check online if needed.

# Further installation for Python 3 version (tested on MacOS and Linux)
1. Download a Python 3 distribution package or for MacOS you can use the native iPython in Terminal. I recommend the Anaconda distribution for anything Python related in general [https://docs.anaconda.com/anaconda/install/].
2. Download the Python script `parasyncstitchicGM.py` and `requirements.txt` and put them into a folder of your choice.
3. If on macOS/Linux, open terminal, navigate (using `cd` unix command) to the folder above and run `pip install -r requirements.txt`. If this doesn't work, then open up the requirements file and manually install the packages listed in the file using `pip install package-name` command. For Windows, the anaconda distribution should come with a command line tool which will allow you to install the required packages in a similar manner.
4. Open the file `parasyncstitchicGM.py` in a script editor (Anaconda provides Spyder as a very useful script editor and execution tool for this) and search for the beginning of the main function on line 92 where you should see multiple lines with `get_platform()`. This is used to check which platform you are on as file paths will be different between them. For your operating system choice, change the `imagejpath` and `overlapypath` file paths for your own file paths to those files.

# Executing Stitching on Python
This pipeline is written to be executed alongside TissueCyte acquisition and can be run as soon as a TissueCyte scan is started. The script will wait until all tiles per section is generated (and will tell you its status in the output dialogue), and then perform the stitching. In this manner, stitching can be completed soon after acquisition is complete to massively expedite initial processing steps. The script can also be run post hoc as all tiles already exist. No changes need to be made to input parameters for this to occur, checking tile existence immediately returns True result.

1. If on macOS/Linux, start Python from Terminal (`ipython` for example), or use your own Python shell from your chosen distribution. Anaconda/Spyder has a terminal built in which can be used to run files.
2. Execute the stitching file using for example `exec(open('parasyncstitchicGM.py').read())`.
3. You should be greeted with a series of lines telling you to fill in the following variables. Press Enter to confirm this, then fill in the requested variables by typing into the terminal command line and pressing Enter. You may drag and drop the folders if possible (not tested on Windows) to make entering the file paths much easier. An example of inputs is shown below. Blank inputs represent choice of the default values. A Parasynchronous Stitching header should be generated after successful completion of the parameter input dialogue. The script will execute by generating the following folder structure residing in the input scan folder `scan-name-Mosaic > channel-number_Stitched_Sections` or additionally `channel-number_Stitched_Sections_Downsized` if downsizing is chosen in the parameter input stage.
```
------------------------------------------
             Parameter Input
------------------------------------------

Fill in the following variables. To accept default value, leave response blank.
Please note this creates a temporary folder to hold images. You require at least 1 GB of free space.
Press Enter to continue:
Select TC data directory (drag-and-drop or type manually): /Volumes/TissueCyte/180517_Hala_5FAD_3738
Start section (default start):
End section (default end): 2
X overlap % (default 1):
Y overlap % (default 1):
Channel to stitch: 2
Perform average correction? (y/n): y
Perform additional downsize? (y/n): n


---------------------------------------------
          Parasynchronous Stitching
---------------------------------------------

Computed average tile.
Stitching Z001...
Complete!
Computed average tile.
Stitching Z002...
Complete!

Stitching completed in 00:00:01:39
```
# Installation specifically for MATLAB version (deprecated)
Download MATLAB if not installed already. It is also worth opening ImageJ/Fiji to start the automatic update procedure. Then download the scripts in the code directory (see [Imperial TissueVision Stitching Files](https://github.com/ImperialCollegeLondon/ImperialTissueCyte/tree/master/Imperial_TissueVision_Stitching_Files))  and place the scripts in a folder which is on your MATLAB path.

# Fiji set-up
These steps enable MATLAB to communicate with Fiji.
1. Install the MIJ plugin, go to `Help > Update Fiji` and click on “Manage update sites” on the bottom left of the window.
2. On the package list which appears, scroll down to "ImageJ-MATLAB" and check the checkbox which is to the left of the package name. Click on “Close” to close the window.
3. In the package window, there should now be a list of packages required to install MIJ. Click on “Apply changes” to install the plugin. Restart Fiji when finished.

## MATLAB set-up
Fiji is now set-up to receive MATLAB communication, but we now need to tell MATLAB to use the correct JAVA environment and how to communicate.
1. The “Grid/Collection stitching” plugin requires Java version 8. Start by installing this for your workstation using the following [link](https://java.com/en/download/).
2. MATLAB now needs to be configured to run in a Java version 8 environment. Go to the following [link](https://uk.mathworks.com/matlabcentral/answers/103056-how-do-i-change-the-java-virtual-machine-jvm-that-matlab-is-using-on-macos) and download the `createMATLABShortcut.m` file. You will find there also exists links/files for *Windows* and *Linux* so download accordingly.
3. In MATLAB, execute the `createMATLABShortcut.m` script which will generate a desktop shortcut called "MATLAB_JVM”.
4. Double-click on "MATLAB_JVM” which will open up MATLAB then confirm that the environment is Java version 8 by typing into the console `version –java` which will return a couple of lines stating the version is 1.8.
5. Confirm MATLAB is able to communicate with Fiji by opening MATLAB via the desktop shortcut and typing in `addpath('/Applications/Fiji.app/scripts');` where the path reflects the path for your system. Follow this with `javaaddpath('/Applications/MATLAB_R2017a.app/java/mij.jar');`, where again the file path is likely specific for where the `mij.jar` file is located on your system.  Finally type in `Miji;` which will return several lines informing that ImageJ/Fiji has been correctly opened within MATLAB.
6. Close down the Fiji communication with `MIJ.exit();`.
7. As Fiji is now being run within MATLAB, memory allocations are restricted by the memory available to MATLAB. Change this by going to `MATLAB > Preferences` then selecting "General" and "Java Heap Memory”. Move the memory slider three-quarters of the way across and apply the change. Close MATLAB.

## Final set-up steps
ImageJ/Fiji and MATLAB are now correctly set-up to communicate with one another. However an issue with *macOS*, which may not be problematic with *Windows* or *Linux* operating systems, is that there exists a limit to the number of open files. Solve this by doing the following.
1. Launch "Terminal" and type `sudo launchctl limit maxfiles unlimited unlimited`.
2. Confirm the change by typing `launchctl limit maxfiles` which will return information where the max file limits are set to a maximum value.
Note that this max file change is only confirmed for the duration the workstation remains turned on. If there is a restart or the system is turned off at any point, the above steps will need to be repeated.

# Executing Stitching on MATLAB
Before starting, make sure the NAS drive or data storage drives are connected to the workstation which will be performing the stitching. This stitching process can be executed immediately following the execution of the TissueCyte scanning procedure.
1. Open MATLAB using the shortcut generated during the installation process to load using the Java version 8 environment.
2. Navigate to the `asyncstitchicGM.m` file downloaded from the directory and run it.
3. The first file browser window which appears requests the path of the root folder where the image data from TissueCyte is being stored. This will be the folder manually created during the set-up for a TissueCyte scan and will contain the sections folders which subsequently contain the raw tile images.
4. The second file browser window which appears requests the path for a temporary folder on the local workstation which can be used to bypass the Bioformats plugin bug associated with NAS drives. If you don’t have an empty folder created yet, use the new folder button to do so. This folder is routinely emptied over the course of the stitching so make sure this folder does not contain any other files.
5. The third window which appears requests the parameters for the scan. These parameters can be found in the `Mosaic.txt` text file which resides in the root folder of the scan. Fill these details in. The overlap percentages can be left as they are but can be changed if desired. Finally, choose the channel you intend to stitch.
The script will now execute by collecting the tile images per physical section and transferring them over to the temporary folder. Here the tiles are averaged together and each individual tile is illumination corrected before being fed into the stitching plugin using ImageJ/Fiji. Each stitched image is then moved back to the data storage drive under a newly generated Mosaic folder in the root directory of the scan. Following this, the temporary folder is emptied and the process repeated. For scans not involving optical sectioning, the stitching for a section completes before imaging so the script will appropriately wait until the corresponding tile images are generated by TissueCyte.

# Things to note
In the parts of the code which compute the average tile and corrects each tile by dividing through by the average, there is a multiplication by 1000. This is to multiply the pixel values of the image such that they cover a greater range of values than they would otherwise occupy if this correction is done. Logically, if you divide an image by a similar image, most values will end up tending towards a value of 1. However, 16-bit images occupy the range 0-65535 so an image with pixel values close to 1 will appear blank, and this is made worse by the fact that converting an image to 16-bit also forces integer values so many pixels will be the same value and you lose detail in the image. Therefore, multiplying the image by 1000 is intended to expand this range so values are no longer close to 1 and detail is maintained when converting the image to 16-bit. You may need to change this value if you discover your images are either still dark, or too bright.
