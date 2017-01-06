# TissueCyte Parameter Generation Program

Part of the TissueCyte operation protocol requires defining the dimensions of the imaging area and how this translates to the number of tiles Orchestrator requires to cover this area during the scan.

In the original protocol this required a laborious process of moving the sample underneath the laser to define the left, right, front and back edge positions. The left-right edge distance is calculated and divided by the FOV to determine the number of X tiles. The same is done for the front-back distance to determine the number of Y tiles. The `setup.exe` supplied above installs a basic program which automatically calcualtes these values including the distance required to move the sample to the microtome cutting position and how to easily reach the front-left corner position from reference.

# Installation

Simply download the `setup.exe` above and run the executable to install the program. 

# Using Program

The program is very basic, think a specialist calculator, and is pretty self-explanatory to use. Simply type in the positions defined by the POS value, only including the sign if the value is negative and formating the value so the last digit is the first decimal place.

Click on `Generate Parameters` to automatically calculate the sample dimensions, distance required to move the sample to the microtome and the movement required to reach the front-left corner of the sample from a newly referenced stage position.

When satisfied, click on `Save Parameters File` to open up a save dialogue to save the parameters to a text file.
