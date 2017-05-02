% Asynchronous stitching (Asyncstitch.m) i.e. stitching on the fly.
% As raw tile images are aquired with TissueCyte, images are immediately
% processed to save on total imageing+stitching time.
%
% Processing applied to each image:
%   - Images are cropped/clipped 1.8% from each edge
%   - Images are illumination corrected according to average illumination 
%   - Images files are renamed and stored into respective directories
%   - Images are loaded and stitched per layer
% 
% Currently only stitches single channel alongside TissueCyte aquisition.
% Script can still be used post imaging for the other channels.

% Clear workspace blah
% clear all
% close all
% clc

% Load up Fiji in MATLAB (without GUI)
Miji(false);

% Collect TC path where raw data is outputted
% This is the directory you create as the save directory when using TC
tcpath = uigetdir('Select directory where TissueCyte will output the raw data');
temppath = uigetdir('Select temp directory');

% Collect relevant variables for TissueCyte scan and processing
prompt={'Scan ID', 'Start section', 'End section', 'Number of X tiles', 'Number of Y tiles', 'Number of Z layers per slice', 'Overlap %', 'Channel to stitch'};
defans={'', '1', '100', '0', '0', '1', '6', '1'}; % Overlap originally 5%
fields = {'id', 'start', 'end', 'xtiles', 'ytiles', 'zlayers', 'overlap', 'channel'};
vars = inputdlg(prompt, 'Please fill in the details', 1, defans);

id = vars{1};
startsec = vars{2};
endsec = vars{3};
xtiles = vars{4};
ytiles = vars{5};
zlayers = vars{6};
overlap = vars{7};
channel = vars{8};

% Create stitch output folders
mkdir([tcpath '/',id,'-Mosaic/Ch',channel,'_Tile_Sections']);
mkdir([tcpath '/',id,'-Mosaic/Ch',channel,'_Stitched_Sections']);

% Variable declare
crop = 0;
filenamestruct = struct();
tstart = tic;
zcount = 1;
filenumber = 0;
tilenumber = 0;
lasttile = -1;

fprintf('        Asynchronous Stitching       \n');
fprintf('-------------------------------------\n\n');

% Check for raw data folders
for section = (str2double(startsec):1:str2double(endsec))
    % Create token for each file
    if section <= 9
        sectiontoken = strcat('000',num2str(section));
    elseif section <= 99
        sectiontoken = strcat('00',num2str(section));
    else
        sectiontoken = strcat('0',num2str(section));
    end
    folder = strcat(id,'-',sectiontoken);
    
    % Token variable hold
    y = str2double(ytiles);
    x = str2double(xtiles);
    xstep = -1;
    
    for layer = (1:1:str2double(zlayers))
        % Check all tiles per layer exist
        completelayer = false;
        %lasttile = (str2double(xtiles)*str2double(ytiles)*layer*section) + (str2double(xtiles)*str2double(ytiles)) - 1;
        %lasttile = lasttile + str2double(xtiles)*str2double(ytiles);
        firsttile = str2double(xtiles)*str2double(ytiles)*(((section-1)*str2double(zlayers))+layer-1);
        lasttile = str2double(xtiles)*str2double(ytiles)*(((section-1)*str2double(zlayers))+layer)-1;
        
        % If last tile doesn't exist yet, wait
        while isempty(dir(strcat(tcpath,'/',folder,'/*-',num2str(lasttile),'_0',channel,'.tif')))
            fprintf('Last tile for layer has yet to be generated. Waiting...\n');
            pause(3);
        end
                
        % When all layer files exist, load each tile for averaging
        for tile = (firsttile:1:lasttile)
            % Get filename string if not already stored
            if isempty(fieldnames(filenamestruct)) == 1
                filenamestruct = dir(strcat(tcpath,'/',folder,'/*-',num2str(filenumber),'_01*'));
            end
            
            if isempty(dir(strcat(tcpath,'/',folder,'/',filenamestruct.name(1:14),num2str(filenumber),'_0',channel,'.tif'))) == 0
                tileimage = double(imread(strcat(tcpath,'/',folder,'/',filenamestruct.name(1:14),num2str(filenumber),'_0',channel,'.tif')));
            else
                tileimage = double(zeros(length(tileimage), length(tileimage)));
            end
                        
            % get crop value if not already stored
            if crop == 0
                crop = round(0.018*length(tileimage));
            end

            % Process image
            % Crop image first around border and rotate
            croplength = length(tileimage)-(2*crop);
            [xsize, ysize] = size(tileimage);
            ycrop = (ysize/2) - (croplength/2);
            xcrop = (xsize/2) - (croplength/2);
            tileimage2 = rot90(imcrop(tileimage, [ycrop xcrop croplength croplength]));
            
            % Sum file to image
            if mod(filenumber+1,str2double(xtiles)*str2double(ytiles)) == 1
                sumimage = tileimage2;
            else
                sumimage = sumimage + tileimage2;
            end
            
            filenumber = filenumber+1;
        end
        
        % Compute average image for layer
        avgimage = sumimage/str2double(xtiles)*str2double(ytiles);
        fprintf('Computed average for current layer\n');
        %imwrite(avgimage, strcat('/Users/gm515/Desktop/Average',num2str(layer),'.tif'));
        
        % Stitch the images per layer together
        % When all layer files exist, load each tile for averaging
        for tile = (firsttile:1:lasttile)
            
            if isempty(dir(strcat(tcpath,'/',folder,'/',filenamestruct.name(1:14),num2str(tilenumber),'_0',channel,'.tif'))) == 0
                tileimage = double(imread(strcat(tcpath,'/',folder,'/',filenamestruct.name(1:14),num2str(tilenumber),'_0',channel,'.tif')));
            else
                tileimage = double(zeros(length(tileimage), length(tileimage)));
            end
                
            % Process image
            % Crop image first around border and rotate
            croplength = length(tileimage)-(2*crop);
            ycrop = (ysize/2) - (croplength/2);
            xcrop = (xsize/2) - (croplength/2);
            tileimage2 = rot90(imcrop(tileimage, [ycrop xcrop croplength croplength]));
            tileimage2 = tileimage2./avgimage;
            
            % TC images in snake pattern starting from bottom left
            % tile. However stitching is done sequentially by row
            % starting from top left position so tiles need to be named
            % according to stitch position, not output number                
            if x>=1 && x<=str2double(xtiles)
                if x < 10
                    xtoken = strcat('00',num2str(x));
                else
                    xtoken = strcat('0',num2str(x));
                end
                x = x+xstep;
                if y < 10
                    ytoken = strcat('00',num2str(y));
                else
                    ytoken = strcat('0',num2str(y));
                end
            elseif x>str2double(xtiles)
                x = str2double(xtiles);
                if x < 10
                    xtoken = strcat('00',num2str(x));
                else
                    xtoken = strcat('0',num2str(x));
                end
                xstep = xstep*-1;
                x = x+xstep;
                y = y-1;
                if y < 10
                    ytoken = strcat('00',num2str(y));
                else
                    ytoken = strcat('0',num2str(y));
                end
            elseif x<1
                x = 1;
                if x < 10
                    xtoken = strcat('00',num2str(x));
                else
                    xtoken = strcat('0',num2str(x));
                end
                xstep = xstep*-1;
                x = x+xstep;
                y = y-1;
                if y < 10
                    ytoken = strcat('00',num2str(y));
                else
                    ytoken = strcat('0',num2str(y));
                end
            end

            if zcount < 10
                ztoken = strcat('00',num2str(zcount));
            elseif zcount < 100
                ztoken = strcat('0',num2str(zcount));
            else
                ztoken = num2str(zcount);
            end
            
            tileimage2 = im2uint16(tileimage2);
            imwrite(tileimage2, strcat(temppath,'/Tile_Z',ztoken,'_Y',ytoken,'_X',xtoken,'.tif'));
                        
            % Stitch the files once all tiles per layer processed
            if mod(tile+1,str2double(xtiles)*str2double(ytiles)) == 0
                % Stitch using Fiji Grid/Collection plugin
                tilepath = strcat(temppath,'/');
                stitchpath = strcat(tcpath,'/',id,'-Mosaic/Ch',channel,'_Stitched_Sections');
                args = strcat('type=[Filename defined position] grid_size_x=',xtiles,' grid_size_y=',ytiles,' tile_overlap=',overlap,' first_file_index_x=1 first_file_index_y=1 directory=',tilepath,' file_names=Tile_Z',ztoken,'_Y{yyy}_X{xxx}.tif output_textfile_name=TileConfiguration_Z',ztoken,'.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory=',stitchpath,'');
                MIJ.run('Grid/Collection stitching', java.lang.String(args));
                MIJ.run('Collect Garbage');
                fclose('all');

                % Rename output file
                file2rename = java.io.File(strcat(stitchpath,'/img_t1_z1_c1'));
                file2rename.renameTo(java.io.File(strcat(stitchpath,'/Stitched_Z',ztoken,'.tif')));
                delete(strcat(temppath,'/*'));
                fprintf('Finished stitching file Stitched_Z%s\n',ztoken);                
                
                zcount = zcount+1;
                y = str2double(ytiles);
                x = str2double(xtiles);
                xstep = -1;
            end            
            tilenumber = tilenumber+1;
        end
    end
end

% Exit ImageJ instance
MIJ.exit();
fprintf('\n-------------------------------------\n');
fprintf('               -fin-                 \n');
telapsed = datestr(toc(tstart)/(24*60*60), 'DD:HH:MM:SS.FFF');
fprintf('Stitching completed in %s\n',telapsed);
