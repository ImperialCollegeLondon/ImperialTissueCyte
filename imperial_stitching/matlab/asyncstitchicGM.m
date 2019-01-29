%============================================================================================
% Asynchronous Stitching Script
% Author: Gerald M
%
% This script pulls the data generated through TissueCyte (or another microscope system) and
% can perfom image averaging correction on the images if requested, before calling ImageJ
% from the command line to perform the stitching. You will need to have the plugin script
% OverlapY.ijm installed in ImageJ in order for the difference in the X and Y overlap to be
% registered. Otherwise the X overlap will be used for both.
%
% Installation:
% 1) Place this script in any folder you want to execute this from
%
% Instructions:
% 1) Run the script in MATLAB
% 2) Fill in the parameters that you are asked for
%    Note: The temporary directory is required to speed up ImageJ loading of the files
%============================================================================================

% Load up Fiji in MATLAB (without GUI)
javaaddpath('/Applications/MATLAB_R2017a.app/java/mij.jar');
javaaddpath('/Applications/MATLAB_R2017a.app/java/ij-1.51n.jar');
addpath('/Applications/Fiji.app/scripts')
Miji(false);
MIJ.run('OverlapY');
MIJ.run('Close All');

% Collect TC path where raw data is outputted
% This is the directory you create as the save directory when using TC
tcpath = uigetdir('Select directory where TissueCyte will output the raw data');
temppath = uigetdir('Select temp directory');

% Check temporary directory is EMPTY
if length(dir(temppath)) ~= 2
    error('Temporary folder is not empty!')
end

% Collect relevant variables for TissueCyte scan and processing
prompt={'Scan ID', 'Start section', 'End section', 'Number of X tiles', 'Number of Y tiles', 'Number of Z layers per slice', 'X Overlap %', 'Y Overlap %', 'Channel to stitch'};
defans={'', '1', '100', '0', '0', '1', '5', '6', '1'}; % Overlap originally 5%, 6%
fields = {'id', 'start', 'end', 'xtiles', 'ytiles', 'zlayers', 'xoverlap', 'yoverlap', 'channel'};
vars = inputdlg(prompt, 'Please fill in the details', 1, defans);

id = vars{1};
startsec = vars{2};
endsec = vars{3};
xtiles = vars{4};
ytiles = vars{5};
zlayers = vars{6};
xoverlap = vars{7};
yoverlap = vars{8};
channel = vars{9};

% Check average correction
avgcorr = questdlg('Perform average correction?');

% Check conversion to JPEG condition
convert = questdlg('Convert channel to JPEGs?');

% Create stitch output folders
mkdir([tcpath '/',id,'-Mosaic/Ch',channel,'_Stitched_Sections']);

switch convert
    case 'Yes'
        mkdir([tcpath,'/',id,'-Mosaic/Ch',channel,'_Stitched_Sections_JPEG']);
        savepath = strcat(tcpath,'/',id,'-Mosaic/Ch',channel,'_Stitched_Sections_JPEG');
end

% Variable declare
crop = 0;
filenamestruct = struct();
tstart = tic;
zcount = ((str2double(startsec)-1)*str2double(zlayers))+1;
filenumber = 0;
tilenumber = 0;
lasttile = -1;
tileimage=0;

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
        firsttile = str2double(xtiles)*str2double(ytiles)*( (str2double(zlayers)*(section-1)) + (layer-1));
        lasttile = (str2double(xtiles)*str2double(ytiles)*( (str2double(zlayers)*(section-1)) + layer)) - 1;
        
        % If last tiles don't exist yet, wait for it...
        if isempty(dir(strcat(tcpath,'/',folder,'/*-',num2str(lasttile),'_01.tif'))) && isempty(dir(strcat(tcpath,'/',folder,'/*-',num2str(lasttile),'_02.tif'))) && isempty(dir(strcat(tcpath,'/',folder,'/*-',num2str(lasttile),'_03.tif')))
            fprintf(strcat('Tile ',num2str(lasttile),' not generated yet. Waiting.'));
            while isempty(dir(strcat(tcpath,'/',folder,'/*-',num2str(lasttile),'_01.tif'))) && isempty(dir(strcat(tcpath,'/',folder,'/*-',num2str(lasttile),'_02.tif'))) && isempty(dir(strcat(tcpath,'/',folder,'/*-',num2str(lasttile),'_03.tif')))
                pause(1);
                fprintf('.');
                pause(1);
                fprintf('.\n');
                pause(1);
                fprintf('\b\b\b');
            end
        end
        
        filenumber = firsttile;
        
        % When all layer files exist, load each tile for averaging
        for tile = (firsttile:1:lasttile)
            % Get filename string if not already stored
            if isempty(fieldnames(filenamestruct)) == 1
                filenamestruct = dir(strcat(tcpath,'/',folder,'/*-',num2str(filenumber),'_01*'));
            end
            
            if isempty(dir(strcat(tcpath,'/',folder,'/',filenamestruct(1).name(1:14),num2str(filenumber),'_0',channel,'.tif'))) == 0
                try
                    tileimage = double(imread(strcat(tcpath,'/',folder,'/',filenamestruct(1).name(1:14),num2str(filenumber),'_0',channel,'.tif')));
                catch ERR
                    if ~isempty(strfind(ERR.message, 'corrupt'))
                        tileimage = double(zeros(length(tileimage), length(tileimage)));
                    end
                end
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
        
        switch avgcorr
            case 'Yes'
                % Compute average image for layer
                avgimage = sumimage/str2double(xtiles)*str2double(ytiles);
                fprintf('\nComputed average for current layer\n');
                %imwrite(avgimage, strcat('/Users/gm515/Desktop/Average',num2str(layer),'.tif'));
        end
        
        tilenumber = firsttile;
        
        % Stitch the images per layer together
        % When all layer files exist, load each tile for averaging
        for tile = (firsttile:1:lasttile)
            
            if isempty(dir(strcat(tcpath,'/',folder,'/',filenamestruct(1).name(1:14),num2str(tilenumber),'_0',channel,'.tif'))) == 0
                try
                    tileimage = (imread(strcat(tcpath,'/',folder,'/',filenamestruct(1).name(1:14),num2str(tilenumber),'_0',channel,'.tif')));
                catch ERR
                    if ~isempty(strfind(ERR.message, 'corrupt'))
                        tileimage = (zeros(length(tileimage), length(tileimage)));
                    end
                end
            else
                tileimage = (zeros(length(tileimage), length(tileimage)));
            end
                
            % Process image
            % Crop image first around border and rotate
            croplength = length(tileimage)-(2*crop);
            ycrop = (ysize/2) - (croplength/2);
            xcrop = (xsize/2) - (croplength/2);
            tileimage2 = double(rot90(imcrop(tileimage, [ycrop xcrop croplength croplength])));
            switch avgcorr
                case 'Yes'
                    tileimage2 = 10000*tileimage2./avgimage;
            end 
                        
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
            
            tileimage2(1,1) = intmax('uint16');
            tileimage2 = uint16(tileimage2);
            imwrite(tileimage2, strcat(temppath,'/Tile_Z',ztoken,'_Y',ytoken,'_X',xtoken,'.tif'));
                        
            % Stitch the files once all tiles per layer processed
            if mod(tile+1,str2double(xtiles)*str2double(ytiles)) == 0
                % Stitch using Fiji Grid/Collection plugin
                tilepath = strcat(temppath,'/');
                stitchpath = strcat(tcpath,'/',id,'-Mosaic/Ch',channel,'_Stitched_Sections');
                args = strcat('type=[Filename defined position] grid_size_x=',xtiles,' grid_size_y=',ytiles,' tile_overlap_x=',xoverlap,' tile_overlap_y=',yoverlap,' first_file_index_x=1 first_file_index_y=1 directory=',tilepath,' file_names=Tile_Z',ztoken,'_Y{yyy}_X{xxx}.tif output_textfile_name=TileConfiguration_Z',ztoken,'.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory=',stitchpath,'');
                MIJ.run('Grid/Collection stitching', java.lang.String(args));
                MIJ.run('Collect Garbage');
                fclose('all');

                % Rename output file
                file2rename = java.io.File(strcat(stitchpath,'/img_t1_z1_c1'));
                file2rename.renameTo(java.io.File(strcat(stitchpath,'/Stitched_Z',ztoken,'.tif')));
                delete(strcat(temppath,'/*'));
                fprintf('Finished stitching file Stitched_Z%s\n',ztoken);
                
                switch convert
                    case 'Yes'
                        I = imread(strcat(stitchpath,'/Stitched_Z',ztoken,'.tif'));
                        I = imresize(I, 0.5);
                        imwrite(im2uint8(I),strcat(savepath,'/Stitched_Z',ztoken,'.jpg'));
                        fprintf('Converted Stitched_Z%s to JPEG\n',ztoken);
                        clear I;
                end
                
                zcount = zcount+1;
                y = str2double(ytiles);
                x = str2double(xtiles);
                xstep = -1;
            end            
            tilenumber = tilenumber+1;
        end
    end
end

fprintf('\n');
% Exit ImageJ instance
MIJ.exit();

fprintf('\n-------------------------------------\n');
telapsed = datestr(toc(tstart)/(24*60*60), 'DD:HH:MM:SS.FFF');
fprintf('Stitching completed in %s\n',telapsed);
fprintf('               -fin-                 \n');