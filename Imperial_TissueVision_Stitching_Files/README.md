# Stitching Protocol

## About Imperial TissueVision Stitching Files
This wiki contains details about the stitching scripts written at Imperial College London to improve upon the original stitching pipeline (see [Original TissueVision Stitching Files]).

The scripts are written in MATLAB and calls a stitching plugin from ImageJ/Fiji. To enable MATLAB to communicate with ImageJ/Fiji, a Java package called MIJ (MATLAB-ImageJ) needs to be installed (see below for installation instructions) and MATLAB also needs to run with a Java version 8 environment. Due to these requirements, there is a sequence of set-up steps that need to be performed before the stitching scripts can be correctly executed. 

As a side note, there exists a bug in ImageJ/Fiji Bioformats import plugin which struggles to import images which reside on a NAS drive. The code therefore has a loophole written in to bypass this issue, but consequently requires at least 1 GB of available space on your local hard drive.

Image stitching is a computationally intensive task, made more so due to the size of the image tiles acquired with TissueCyte. It is recommended to have a workstation with at least 16 GB of RAM. Currently, the scripts have been confirmed to run on *macOS*, but many of the instructions below should have corresponding instructions in a *Windows* operating system environment. 

## Installation
Download MATLAB and ImageJ/Fiji if not installed already. It is also worth opening ImageJ/Fiji to start the automatic update procedure. Then download the scripts in the code directory (see [Imperial TissueVision Stitching Files])  and place the scripts in a folder which is discoverable with MATLAB.

### Fiji set-up
The first step is to get ImageJ/Fiji correctly set-up to allow MATLAB communication. 
1. In Fiji, check the “Grid/Collection stitching” plugin is installed under `Plugins > Grid/Collection stitching`.
2. Next, to install the MIJ plugin, go to `Help > Update Fiji` and click on “Manage update sites” on the bottom left of the window.
3. On the package list which appears, scroll down to "ImageJ-MATLAB" and check the checkbox which is to the left of the package name. Click on “Close” to close the window.
4. In the package window, there should now be a list of packages required to install MIJ. Click on “Apply changes” to install the plugin. Restart Fiji when finished. 
5. Next, the available RAM for Fiji needs to be increased to avoid memory shortage errors during the stitching. Go to `Edit > Options > Memory & Threads…`. The window which opens should contain a “Maximum Memory” field which can edited to a specific value. Change this to no more than three-quarters of your available system RAM. For example, a 32 GB workstation can be set to have 24000 MB of memory available for Fiji. Avoid exceeding the three-quarter rule as this can make your workstation unstable due to other background processes.
6. Fiji should now be restarted then subsequently closed down. 

### MATLAB set-up
Fiji is now set-up to receive MATLAB communication, but we now need to tell MATLAB how to start communicating.
1. The “Grid/Collection stitching” plugin requires Java version 8. Start by installing this for your workstation using the following [link].
2. MATLAB now needs to be configured to run in a Java version 8 environment. Go to the following [link] and download the `createMATLABShortcut.m` file. You will find there also exists links/files for *Windows* and *Linux* so download accordingly. 
3. In MATLAB, execute the `createMATLABShortcut.m` script which will generate a desktop shortcut called "MATLAB\_JVM”. 
4. Double-click on "MATLAB\_JVM” which will open up MATLAB then confirm that the environment is Java version 8 by typing into the console `version –java` which will return a couple of lines stating the version is 1.8.
5. To allow MATLAB to communicate with Fiji, add the path of the Fiji plugins to MATLAB by typing in `addpath('/Applications/Fiji.app/scripts');` followed by `savepath;`. This is assuming the Fiji application resides in the Applications folder. Change this path accordingly. For *Windows* the script path will likely be different to check beforehand.
6. Confirm MATLAB is able to communicate with Fiji by opening MATLAB via the desktop shortcut and typing in `Miji;` which will return several lines informing that ImageJ/Fiji has been correctly opened within MATLAB.
7. Close down the Fiji communication with `MIJ.exit();`.
8. As Fiji is now being run within MATLAB, memory allocations are restricted by the memory available to MATLAB. Change this by going to `MATLAB > Preferences` then selecting "General" and "Java Heap Memory”. Move the memory slider three-quarters of the way across and apply the change. Close MATLAB.

### Final set-up steps
ImageJ/Fiji and MATLAB are now correctly set-up to communicate with one another. However an issue with *macOS*, which may not be problematic with *Windows* or *Linux* operating systems, is that there exists a limit to the number of open files. Solve this by doing the following. 
1. Launch "Terminal" and type `sudo launchctl limit maxfiles unlimited unlimited`. 
2. Confirm the change by typing `launchctl limit maxfiles` which will return information where the max file limits are set to a maximum value.
Note that this max file change is only confirmed for the duration the workstation remains turned on. If there is a restart or the system is turned off at any point, the above steps will need to be repeated.

## Executing Stitching
Before starting, make sure the NAS drive or data storage drives are connected to the workstation which will be performing the stitching. This stitching process can be executed immediately following the execution of the TissueCyte scanning procedure.
1. Open MATLAB using the shortcut generated during the installation process to load using the Java version 8 environment.
2. Navigate to the `asyncstitchic.m` file downloaded from the directory and run it.
3. The first file browser window which appears requests the path of the root folder where the image data from TissueCyte is being stored. This will be the folder manually created during the set-up for a TissueCyte scan and will contain the sections folders which subsequently contain the raw tile images.
4. The second file browser window which appears requests the path for a temporary folder on the local workstation which can be used to bypass the Bioformats plugin bug associated with NAS drives. If you don’t have an empty folder created yet, use the new folder button to do so. This folder is routinely emptied over the course of the stitching so make sure this folder does not contain any other files.
5. The third and final window which appears requests the parameters for the scan. These parameters can be found in the Mosaic text file which resides in the root folder of the scan. Fill these details in. The overlap percentage can be left at 6% but can be changed if desired. lastly choose the channel you intend to stitch.
The script will now execute by collecting the tile images per physical section and transferring them over to the temporary folder. Here the tiles are averaged together and each individual tile is illumination corrected before being fed into the stitching plugin using ImageJ/Fiji. Each stitched image is then moved back to the data storage drive under a newly generated Mosaic folder in the root directory of the scan. Following this, the temporary folder is emptied and the process repeated. For scans not involving optical sectioning, the stitching for a section completes before imaging so the script will appropriately wait until the corresponding tile images are generated by TissueCyte. 

Further steps can be performed to downsize the images by 50% and convert to JPEGs using the script `tiff2jpegfastGM.m`. To save time, the script runs the conversion process in parallel using the available cores on the workstation.
1. Run the script and navigate to the directory containing the stitched images. Highlight the number of images you want to convert by clicking on the first image, holding `shift-key` then clicking on the last image.
2. Next choose the directory you want to output the converted images then let the script execute. 
