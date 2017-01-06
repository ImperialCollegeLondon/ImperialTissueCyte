# Original TissueVision Stitching Files

This directory contains the original files supplied by TissueVision for stitching the output data from TissueCyte.

Files appended by `.m` require [MATLAB](https://uk.mathworks.com/products/matlab.html) for execution whilst the files appended by `.ijm` are plugins requiring [Imagej/Fiji](http://fiji.sc).

The stitching process is both processing and time intensive. The system being used to run these scripts should be free from other processing heavy tasks and should ideally boast at least 16GB of RAM. Both **macOS** and **Windows** operating systems can be used, but see below for a slight difference in installation instructions. The scripts are not without bugs, and errors caused will terminate the process. Restarting the stitching from where you left off is not trial task so make sure the workstation being used for stitching is in the optimum condition to avoid errors and time wasting. This means checking the storage device containing the raw TissueCyte data is connected (and not likely to lose connection) and contains enough free storage. Processing involves generating new tile images from the raw image data (done with MATLAB) and then goes on to stitch the images together (done with ImageJ/Fiji). Hence the amount of data at the end of the stitching process is roughly tripled so ensure there is storage space available.

Currently the scripts do not have any compatability with parallel copmuting, which would tremendously reduce the processing time, so channels are processed in a one-by-one fashion. You can aid the required time by sitching the channels of interest only, or executing the stitching across several workstations if they are available, each dedicated to a single channel.

# Installation

Download MATLAB and ImageJ/Fiji if you don't already have it, then download everything in this code directory.

## MATLAB

Copy all the MATLAB scripts (ending in `.m`) into a directory which is discoverable by MATLAB. A good habit is to have a dedicated directory which contains all your MATLAB scripts, containing subdirectories if needed. This means only one directory needs to be added to the MATLAB path. 

If you are new to MATLAB, you can accomplish this by simply creating a new folder with an approriate name such as "MATLAB scripts" and copying the files into this folder. Open MATLAB and on the left pane will be the root directory of MATLAB. Within the pane, navigate to the folder *containing* your MATLAB scripts folder, right click on the folder and choose `Add to path > Selected Folders and Subfolders`.

For those running **macOS**, the MATLAB installation steps are complete. To test everything works, in the MATLAB command line terminal, type in `stitch` and press enter which should open a dialogue window to start the stiching process.

For those running **Windows**, in the MATLAB scripts folder is a file named `main-program.bat`. Right click on it and create a shortcut to the desktop. Double-click the shortcut to start stitching with MATLAB.

## ImageJ/Fiji

Next, copy the `.ijm` files inside "ImageJ-Plugins" and paste them into a suitable directory. Open Fiji and select `Plugins > Install plugin...` then navigate to the "ImageJ-Plugins" folder and select any of the three plugin files. Repeat the installation process for the other two plugin files. When Fiji is closed and reopened, the plugins you installed should be listed under the `Plugins` menu. Clicking on either of the installed plugins should open a dialogue box to start the Fiji stitching process.

# Running Stitching Protocol

The original protocol for running the stitching pipeline using MATLAB and Fiji can be seen [here](ImperialTissueCyte/Original_TissueVision_Stitching_Files/Stitching_Protocol_Original.pdf).
