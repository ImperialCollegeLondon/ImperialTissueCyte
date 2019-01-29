%% Clears all previous open windows and/or variables
clear all
close all
clc

%% Prepares image set for stitching
% Replace the "param_file_name" variable with the directory path and main mosaic file name
% Change the number of sections based on how many were taken
% Set special case to 1 if the file names that the code attempting to open
% are wrong, MATLAB will prompt you to enter new values.


[filename, pathname] = uigetfile('*.*','Select the Mosaic file to begin:');

averageq = questdlg('Would you like to perform flat field correction?','Flat field correction?','Yes','No','Cancel','No');

questoutput = questdlg('Would you like to output the files in a specified directory?','Output directory specification?','Yes','No','Cancel','No');

switch questoutput
    case 'Yes'
        [outputdir] = uigetdir('*.*','Select the output directory')
        
        if ischar(outputdir)==0
            disp('You have canceled the operation.');
            return;
        end
    case 'No'
        outputdir=0
    case 'Cancel'
        disp('You have canceled the operation.');
        return;
end

switch averageq
    case 'Yes'
        averageanswr=1;
    case 'No'
        averageanswr=0;
    case 'Cancel'
        disp('You have canceled the operation.');
        return;
end

if ischar(filename)==0
    disp('You have canceled the operation.');
    return;
end

prompt={'Starting section', 'Ending section', 'First section for image average generation','Last section for image average generation', 'Number of Channels to include'};
defans={'1', '100', '1','100','3'};
fields = {'start','end', 'avgstart','avgend','channels'};
vars = inputdlg(prompt, 'Please provide these variables:', 1, defans);
if ~isempty(vars)
    vars = cell2struct(vars,fields);
else
    disp('You have canceled the operation.');
    return;
end
vars.spec = '1';
param_file_name = sprintf('%s%s',pathname,filename)
start = str2num(vars.start)
ending = str2num(vars.end)
avgstart = str2num(vars.avgstart)
avgend = str2num(vars.avgend)
chan = str2num(vars.channels)
tstart = tic;


Export2Fiji_reverse(param_file_name, [start,ending], [avgstart,avgend], [1,chan],outputdir,averageanswr);
telapsed = toc(tstart);
disp('Time Elapsed:');
disp(telapsed);

