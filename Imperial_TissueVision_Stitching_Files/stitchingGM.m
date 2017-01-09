% Script to perform stitching pipeline on image data
%
% Stitching is performed with the Grid/Collection plugin in ImageJ/Fiji.
% Whilst the plugin is much faster than the depracted versions, it uses 
% Bioformats to import data which has a bug which causes it to import
% images from a NAS drive at very slow speed. TO circumvent this, the
% script copies the files needed to stitch a slice into a local temporary
% folder allowing Bioformats to import the images at normal speed.

% Clear all
clear all
close all
clc

% Load up Fiji in MATLAB (without GUI)
Miji(false);
fprintf('ImageJ instance loaded cleanly\n');
    
% Collect relevant directories
tilepath = uigetdir('Select directory containing tile files');
stitchpath = uigetdir('Select directory to output stitched files');
temppath = uigetdir('Select directory for temporary file storage');

% Collect relevant variables for TissueCyte scan
prompt={'Start section', 'End section', 'Number of X tiles', 'Number of Y tiles', 'Number of Z layers per slice', 'Overlap %', 'Channel to stitch'};
defans={'1', '100', '0', '0', '1', '7', '1'}; % Overlap originally 5%
fields = {'start','end', 'xtiles', 'ytiles', 'zlayers', 'overlap', 'channel'};
vars = inputdlg(prompt, 'Please fill in the details', 1, defans);

if isempty(vars)
    return;
end
    
startsec = vars{1};
endsec = vars{2};
xtiles = vars{3};
ytiles = vars{4};
zlayers = vars{5};
overlap = vars{6};
channel = vars{7};

fprintf('Starting stitching process\n');
tstart = tic;

% Loop through each layer and stitch with Fiji
for i = ((str2double(startsec)-1)*str2double(zlayers))+1:1:str2double(zlayers)*str2double(endsec)
    if i <= 9
        zcount = strcat('00',num2str(i));
    elseif i <= 99
        zcount = strcat('0',num2str(i));
    else
        zcount = num2str(i);
    end
    
    % Copy files into temporary folder
    system(['cp -a ' strcat(tilepath,'/Tile_Z',zcount,'*') ' ' temppath]);
    
    % Stitch using Fiji Grid/Collection plugin
    args = strcat('type=[Filename defined position] grid_size_x=',xtiles,' grid_size_y=',ytiles,' tile_overlap=',overlap,' first_file_index_x=1 first_file_index_y=1 directory=',temppath,' file_names=Tile_Z',zcount,'_Y{yyy}_X{xxx}.tif output_textfile_name=TileConfiguration_',zcount,'.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory=',temppath,'');
    MIJ.run('Grid/Collection stitching', java.lang.String(args));
    MIJ.run('Collect Garbage');
    fclose('all');
    
    % Rename output file and delete temp files
    system(['mv ' strcat(temppath,'/img_t1_z1_c1') ' ' strcat(stitchpath,'/Stitched_Z',zcount,'.tif')]);
    delete(strcat(temppath,'/*'));
    fprintf('Finished stitching file Stitched_Z%s\n',zcount);
end

% Exit ImageJ instance
MIJ.exit();
telapsed = datestr(toc(tstart)/(24*60*60), 'DD:HH:MM:SS.FFF');
fprintf('Stitching completed in %s\n',telapsed);