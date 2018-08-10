# About Imperial TissueVision Stitching Files
**Update 09/08/18** The Grid/Collection stitching plugin has a secret parameter which can be added allowing the overlap in X and Y to be separately defined. This is enabled by using the `OverlapY.ijm` file which is a bean shell file that can be run ImageJ/Fiji to enable that option.

This wiki contains details about the stitching scripts written at Imperial College London to improve upon the original stitching pipeline (see [Original TissueVision Stitching Files](https://github.com/ImperialCollegeLondon/ImperialTissueCyte/tree/master/Original_TissueVision_Stitching_Files)).

The scripts are written in both MATLAB and Python and calls a stitching plugin from ImageJ/Fiji. To enable MATLAB to communicate with ImageJ/Fiji, a Java package called MIJ (MATLAB-ImageJ) needs to be installed (see below for installation instructions) and MATLAB also needs to run with a Java version 8 environment. Due to these requirements, there is a sequence of set-up steps that need to be performed before the stitching scripts can be correctly executed. The Python version of the script should run much easier without having to do any changes other than install the packages in the requirements file.

As a side note, there exists a bug in ImageJ/Fiji Bioformats import plugin which struggles to import images which reside on a NAS drive. The code therefore has a loophole written in to bypass this issue, but consequently requires at least 1 GB of available space on your local hard drive.

Image stitching is a computationally intensive task, made more so due to the size of the image tiles acquired with TissueCyte. It is recommended to have a workstation with at least 16 GB of RAM. Currently, the scripts have been confirmed to run on *macOS*, but many of the instructions below should have corresponding instructions in a *Windows* operating system environment.

The Grid/Collection stitching plugin has a secret parameter which can be added allowing the overlap in X and Y to be separately defined. This is enabled by using the `OverlapY.ijm` file which is a bean shell file that can be run ImageJ/Fiji to enable that option.

# Installation required for both MATLAB and Python
The following is required for both the MATLAB and Python version of the sctitching pipeline.

## Fiji set-up
The first step is to get ImageJ/Fiji correctly set-up to allow MATLAB communication.
1. In Fiji, check the “Grid/Collection stitching” plugin is installed under `Plugins > Grid/Collection stitching`.
2. Next, to install the MIJ plugin, go to `Help > Update Fiji` and click on “Manage update sites” on the bottom left of the window.
3. On the package list which appears, scroll down to "ImageJ-MATLAB" and check the checkbox which is to the left of the package name. Click on “Close” to close the window.
4. In the package window, there should now be a list of packages required to install MIJ. Click on “Apply changes” to install the plugin. Restart Fiji when finished.
5. Next, the available RAM for Fiji needs to be increased to avoid memory shortage errors during the stitching. Go to `Edit > Options > Memory & Threads…`. The window which opens should contain a “Maximum Memory” field which can edited to a specific value. Change this to no more than three-quarters of your available system RAM. For example, a 32 GB workstation can be set to have 24000 MB of memory available for Fiji. Avoid exceeding the three-quarter rule as this can make your workstation unstable due to other background processes.
6. Fiji should now be restarted then subsequently closed down.
7. Finally, download the `OverlapY.ijm` file and move it to the Plugins folder of your ImageJ/Fiji application (on MacOS this is found by right clicking the application and showing package contents).

# Installation specifically for MATLAB version
Download MATLAB if not installed already. It is also worth opening ImageJ/Fiji to start the automatic update procedure. Then download the scripts in the code directory (see [Imperial TissueVision Stitching Files](https://github.com/ImperialCollegeLondon/ImperialTissueCyte/tree/master/Imperial_TissueVision_Stitching_Files))  and place the scripts in a folder which is discoverable with MATLAB.

## MATLAB set-up
Fiji is now set-up to receive MATLAB communication, but we now need to tell MATLAB how to start communicating.
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

# Installation for specifically Python version (tested on MaCOS)
1. Download a Python distribution package or for MacOS you can use the native iPython in Terminal.
2. Download the Python scripts and put them into a folder of your choice.
3. From Terminal, navigate to the folder and run `pip install -r requirements.txt`. If this doesn't work, then open up the requiremnts file and manually install the packages listed there.
4. Create a temporary folder on your hard drive which will temporarily store image data during the stitching process.
5. Open the `asyncstitchicGM.py` file and go to line 222 which begins with a `subprocess.call`. Change the file path for ImageJ-macosx to the file path for your system. It should be reasonably similar to the one already there. Also change the file path for the OverlapY.ijm file for the path on your system.

# Executing Stitching on Python
1. In Terminal, run Python such as `ipython` or use your own Python shell from your distribution.
2. Run the `asyncstitchicGM.py` file.
3. Each line which appears asks for the same parameters see in Executing Stitching on MATLAB above.

# Things to note
In line 168 of the Python `asyncstitchicGM.py` script and line 190 of the MATLAB `asyncstitchicGM.m` script there is a multiplication by 10000. This is to multiply the pixel values of the image such that cover a greater range of pixel values than they would otherwise occupy. Typically, most images will look completely blank and will require contrast adjustments in ImageJ/Fiji to see the data. There is data there - don't worry, but its hidden behind the range of 16-bit values you can take. Multiplying the image by 10000 is intended to expand this range so the image is more easily seen. Not every image will require multiplication by this value so adjust this value as necessary depending on how your images look. You could ommit this multiplication if you want, but if you perform image averaging correction, note that you massively reduce the intensity range of your image as average correction involves dividing the image.