# Imperial TissueVision Stitching Files

This directory contains scripts written to improve upon the original stitching pipeline (see [Original TissueVision Stitching Files](https://github.com/gm515/ImperialTissueCyte/tree/master/Original_TissueVision_Stitching_Files)). As it stands, the MATLAB portion of the original stitching pipeline is still required to process the raw image tile data. The scripts here instead replace the Fiji portion of the pipeline using MATLAB, but also integrating the default Fiji plugin "Grid/Collection stitching".

Currently, there exists a bug in the Fiji plugin which makes importing files stored on a NAS drive a tediously slow and an essentially broken process. A workaround for this has been implemented into the scripts, but despite this, execution time for the stitching is less than a third of the time taken in the original process. Once parallelisation has been added in the future, this will massively reduce execution time.

Fiji is called internally with MATLAB using the Java package [MIJ](http://bigwww.epfl.ch/sage/soft/mij/). To run without any errors, this can require a few additional installation steps and also requires MATLAB to be running with a Java version 8 environment. To get everything set up for executing these scripts, follow the steps bellow. 

As stitching is a computing intensive process, it is recommended to use a computer with at least 16GB of RAM. This code has also been tested on a **macOS** system only, but there is nothing to prohibit the scrips being run on a **Windows** system. Some of the steps for making MATLAB compatible with MIJ will of course be operating system dependant but corresponding steps should exist online for **Windows**. 

**UPDATE** Asynchronous stitching (stitching as TissueCyte generates files) has now been created in a separate stitching script with MATLAB. The script to download for this is called _asyncstitchic.m_. The script works by determining which file is expected to be created by TissueCyte at the end of each imaged layer. When the final tile file is created, the script loads up all the tiles for a single layer then calculates the average image. Each individual tile is then loaded, cropped and rotated as necessary, then divided through by the average tile to correct for intensity differences across a single tile. Each of these processed tiles is saved into a Mosaic folder. Once the tiles for a single layer are processed, Fiji is called to start the stitching process. The script will do this for all the expected sections and layers in the scan (user fed when running the script). In such a way, the asynchronous stitching algorithm is capable of massively reducing the time it takes to image _and_ stitch data as the process is carried out in parallel. See the updated protocol below to see how to run this process.

# Installation

Download MATLAB and ImageJ/Fiji if you don't already have it, then download the two scripts in this code directory and place them into folder added to the MATLAB path.

## Fiji

In Fiji, check the "Grid/Collection stitching" exists under `Plugins > Grid/Collection stitching`. It should already be installed as default. To install the MIJ package, go to `Help > Update Fiji` which will open the "ImageJ Updater" window. Click on "Manage update sites" on the bottom left, find the package "ImageJ-MATLAB" and check the checkbox on the left of the name. Close "Manage update sites" window and you'll see the necessary packages to install are listed in the updater window. Go ahead and click "Apply changes" to install the package. Restart Fiji if required.

Next you want increase the RAM memory available to Fiji to avoid errors with memory shortage. More RAM also allows the stiching plugin to execute faster whilst being slightly memory inefficient. Go to `Edit > Options > Memory & Threads...`. In the window that opens, you will see a Maximum Memory value which will be set to MB value. Change this value to no more than three-quarters of your total system memory. For example, a system with 32GB of RAM, change the maximum memory value to 24000MB. Do not exceed the three-quarters rule as this can make your system unstable. Remember, that the more RAM allocated to Fiji, the less there is available for other system processes!

Fiji should now be correctly configured, so close it down. When executing stitching scripts, you won't actually need to open Fiji yourself.

## Setting up MATLAB to run Fiji

The Grid/Collection stitching plugin requires Java version 8, but by default MATLAB only uses Java version 7. First install [Java version 8](https://java.com/en/download/) if not already installed on your system. 

Next go to the following [link](http://uk.mathworks.com/matlabcentral/answers/103056-how-do-i-change-the-java-virtual-machine-jvm-that-matlab-is-using-for-mac-os) and download the file `createMATLABShortcut.m`. If you are on **Windows** you can also find the relevent solution for creating a Java version 8 MATLAB environment with one of the links on the page. Open MATLAB and run the `createMATLABShortcut.m` script which will generate a "MATLAB_JVM" shortcut on your desktop. Double-click the shortcut and MATLAB using Java 8 should open. Confirm this by typing in `version –java` into the command terminal, which should return the version as 1.8.

To allow MATLAB to access the MIJ plugin you downloaded in Fiji earlier, in the command terminal, type in
```matlab
addpath('/Applications/Fiji.app/scripts');
savepath;
```
where `'/Applications/Fiji.app/scripts'` is the path to the installed scripts within Fiji. The path will be correct if Fiji is installed in the default "Applications" directory. If not, change the path accordingly. Confirm MIJ is set up by typing
```matlab
Miji;
```
which should return dialogue confirming an ImageJ/Fiji instance has been successfully loaded. Close the instance down with
```matlab
MIJ.exit();
```

As Fiji is now being run inside MATLAB, memory allocation is dependant on the memory of MATLAB itself. Maximise the MATLAB memory through `MATLAB > Preferences` then selecting "General" and "Java Heap Memory". Move the memory slider all the way to the right, apply the change then close down MATLAB. The MATLAB environment is now correctly configured for the scripts, however one more change is required, as below.

To finish the MATLAB configuration, you need to increase the system memory allocated for the number of open files you can have open. Open "Terminal" and type
```unix
launchctl limit maxfiles
```
to check how much memory is allocated for open files. Change this by typing the following
```unix
sudo launchctl limit maxfiles unlimited unlimited
```
to set the number of allowed open files to their maximum. You can check the change is confirmed by typing in
```unix
launchctl limit maxfiles 
```
again and comparing the new values to the old. This open files limit change is only temporary for the time the computer system is on. If the system is turned off or restarted, then you will need to redo this process.

# Running Stitching Protocol

Open up MATLAB and type in `stitchingGM` into the command terminal. The script should be located in a folder linked to the MATLAB path so it should immediately start without errors. Three dialogue windows should open asking for a particular directory. The first window asks for the folder of tiled images generate through MATLAB using the original stitching pipeline. This will be one of the channel folders with the name containing "Ch#\_Tiled_Sections_" where # indicates the channel number. The second window asks for the output folder where the stiched files will be saved. You can use the stitched folders generated by the original MATLAB step, in which case the output folder will contain the name "Ch#\_Stitched_Sections_". Finally, the third window asks for a temporary folder where the tiles per layer can copied. This temporary folder is neccessary to bypass the bug in the Grid/Collection stitching plugin. The temporary folder should be located on your local drive. All files copied to the folder are automatically deleted when used so you don't need to worry about having a lot of space available. However, there should atleast be enough space to hold all the tiles for a single layer only, roughly 100 files.

Once the folders have been chosen, the following dialogue window asks for parameters about the scan itself. Fill in the relevant parameter values start the stitching process. MATLAB should start copying the files for a single section to the temporary folder, loading Fiji and running the stitching plugin. After a single layer is stitched, the stitched image is copied to the output directory and the files in the temporary folder are deleted. The process is then repeated for the remaining sections. 

Once all the layers are stitched, the script should end and print out information about the execution time. The output folder chosen earlier will contain all the stitched files in raw dimensions. Of course this means each image can be of the order of several hundred MB. The "tiff2jpegGM" script can be used to downsize the images and save them as JPEGs to make them more manageable to handle. Simply type in `tiff2jpegGM` into the MATLAB command terminal. In the following window that opens select the files you want to convert. Hold shift on the keyboard to select a range of files. MATLAB will then use Fiji to lead each image, downsize, enhance contrast and save as a JPEG into the same folder as the stitched output files.

**UPDATE** To use the asynchronous stitching algorithm with MATLAB, download _asyncstitchic.m_ and place into a directory which is part of the MATLAB path. In MATLAB, execute the script. In the window that opens, fill in the relevant scanning details and the channel you wish to stitch. The algorithm should immediately execute and will update you on progress for when a layer has been averaged and then finally stitched. If the necessary files have yet to be generated by TissueCyte, the script will stall itself and wait until the files are generated. Hence the alogrithm can be left to carry out the stitching by itself.
