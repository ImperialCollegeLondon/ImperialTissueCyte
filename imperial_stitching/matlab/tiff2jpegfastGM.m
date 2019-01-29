% Apply 'enhance contrast' and save tiff as jpeg
% Executed in parallel

% get list of files to convert
[filenames,pathname] = uigetfile('*.tif','Select TIFF files to convert','MultiSelect','on');
savepath = uigetdir('Select directory to save JPEG files');

disp(pathname);
disp(filenames);

tstart = tic;

% convert TIFF to JPEG
parfor i=1:1:length(filenames)
    [pathstr,name,ext] = fileparts(char(filenames(i)));
    I = imread(strcat(pathname,name,ext));
    I = imresize(I, 0.5);
    imwrite(im2uint8(I),strcat(savepath,'/',name,'.jpg'));
    fprintf('Saved image %s.jpg\n',name);
end

% wrap up and finish
fprintf('Finished TIFF conversion to JPEG\n');
telapsed = datestr(toc(tstart)/(24*60*60), 'DD:HH:MM:SS.FFF');
fprintf('Stitching completed in %s\n',telapsed);