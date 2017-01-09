% Apply 'enhance contrast' and save tiff as jpeg

% get list of files to convert
[filenames,pathname] = uigetfile('*.tif','Select TIFF files to convert','MultiSelect','on');
savepath = uigetdir('Select directory to save JPEG files');

disp(pathname);
disp(filenames);

% load up Fiji in MATLAB (without GUI)
Miji(false);
fprintf('ImageJ instance loaded cleanly\n');

tstart = tic;

% enhance contrast and save as jpeg
for i=1:1:length(filenames)
    savename = strrep(filenames(i),'.tif','.jpg');
    argopen = strcat('path=[',pathname,filenames(i),']');
    MIJ.run('Open...',java.lang.String(argopen));
    MIJ.run('Enhance Contrast...', 'saturated=0.3');
    MIJ.run('Scale...', 'x=0.5 y=0.5 width=11480 height=9575 interpolation=Bilinear average create');
    pause(1);
    argsave = strcat('path=[',savepath,'/',savename,']');
    MIJ.run('Jpeg...',java.lang.String(argsave));
    MIJ.run('Close All');
    
    fprintf('Saved image %s\n',savename{1});
end

% wrap up and finish
MIJ.exit();
fprintf('Finished TIFF conversion to JPEG\n');
telapsed = datestr(toc(tstart)/(24*60*60), 'DD:HH:MM:SS.FFF');
fprintf('Stitching completed in %s\n',telapsed);