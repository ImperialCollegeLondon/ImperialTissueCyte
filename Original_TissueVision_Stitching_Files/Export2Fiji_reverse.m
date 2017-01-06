%--------------------------------------------------------------------------
% Export2Fiji_reverse.m script is used to convert tiff files from TissueCyte to be used with Fiji's stitching plugin.
%
% Usage
% param = Export2Fiji_reverse('/media/Tissue Vision 1/Salter Brain 1/Mosaic_Salter_Microglia_Brain_1_Nov222011.txt', [100,100], [1,100], [1,3])
%
% History
% Author   | Date         |Change
%==========|==============|=================================================
% Wong     | 2011 Dec 00  |Initial Creation
% Dazai    | 2011 Dec 00  |created the reverse order to account for
%          |              |photobleaching.  Added Illumination
%          |              |correction.
% Dazai    | 2012 Jan 10  |Fixed file numbering error and added 3 channel
%          |              |capability.
% Dazai    | 2012 Jan 11  |Able to read in multiple layers per slice.
%--------------------------------------------------------------------------
function Export2Fiji_reverse(param_file_name,SectionInfo,AveragingSectionInfo,ChannelInfo,outputdir,averageanswr)

[directory,name,extension]=fileparts(param_file_name);

param_file_ptr = fopen (param_file_name);
tokeniser = ':';
tokeniser2 =' ';
tokeniser3 ='_';
tokeniser4 ='-';

Initial_Section=SectionInfo(1);
Final_Section=SectionInfo(2);
Initial_Averaging=AveragingSectionInfo(1);
Final_Averaging=AveragingSectionInfo(2);
Initial_Channel=ChannelInfo(1);
Final_Channel=ChannelInfo(2);


param = struct('fileName', param_file_name);



n = 0;
line = 0;
while ((~(line == -1)) & (n < 38))
    line = fgetl (param_file_ptr);
    
    % This is a really gross hack to parse out the acqDate correctly
    % The existing code eats the space between the date and the 
    % time.  I want to keep this and not change the existing code
    % becasue it is pretty fragile
    if( (length(line) >= 7)  && strcmp(line(1:7),'acqDate'))
        
        param.acqDate = line(9:end);
       continue
    end
    
    if strfind(line,tokeniser2)>0
        line_space = line(1:(strfind (line, tokeniser2) - 1));
        line_space2 = line((strfind (line, tokeniser2) + 1):end);
        line = [line_space line_space2];
    end
    
    paramName = line (1:(strfind (line, tokeniser) - 1));
    paramValue = line ((strfind (line, tokeniser) + 1):end);
    param = setfield (param, paramName, paramValue);
    nToken = length (strfind (line, tokeniser)) + 1;
    n = n + 1;
end

param.rows = str2num (param.rows);
param.columns = str2num (param.columns);

% Setting numbers values as numbers
param.mrows = str2num (param.mrows);
param.mcolumns = str2num (param.mcolumns);
param.sections = str2num (param.sections);

param.sectionres = str2num (param.sectionres);
param.mrowres = str2num (param.mrowres);
param.mcolumnres = str2num (param.mcolumnres);

param.startnum = str2num (param.startnum);
param.layers = str2num (param.layers);
param.xres = str2num (param.xres);
param.yres = str2num (param.yres);
param.zres = str2num (param.zres);
param.channels = str2num (param.channels);
param.Pixrestime = str2num (param.Pixrestime);
param.Zscan = str2num (param.Zscan);
% param.xoverlap = str2num (param.xoverlap);
% param.yoverlap = str2num (param.yoverlap);
% param.zoverlap = str2num (param.zoverlap);
param.paramFileName = param_file_name;

clip = round(param.rows*0.018);
clipwidth = [clip clip clip clip];
param.clipwidth = clipwidth;

file_name_timestamp = find_file_name_base(param);

if param.Zscan==0
    param.layers=1;
end

for folder=Initial_Channel:Final_Channel
    
    if outputdir == 0
        mkdir([directory '/',param.SampleID,'-Mosaic/Ch',num2str(folder),'_Tile_Sections_',num2str(Initial_Section),'_to_',num2str(Final_Section)]);
        mkdir([directory '/',param.SampleID,'-Mosaic/Ch',num2str(folder),'_Stitched_Sections_',num2str(Initial_Section),'_to_',num2str(Final_Section)]);
        %mkdir([directory '/',param.SampleID,'-Mosaic/Ch',num2str(folder),'_Aligned_Sections_',num2str(Initial_Section),'_to_',num2str(Final_Section)]);
    else
        mkdir([outputdir '/',param.SampleID,'-Mosaic/Ch',num2str(folder),'_Tile_Sections_',num2str(Initial_Section),'_to_',num2str(Final_Section)]);
        mkdir([outputdir '/',param.SampleID,'-Mosaic/Ch',num2str(folder),'_Stitched_Sections_',num2str(Initial_Section),'_to_',num2str(Final_Section)]);
        %mkdir([outputdir '/',param.SampleID,'-Mosaic/Ch',num2str(folder),'_Aligned_Sections_',num2str(Initial_Section),'_to_',num2str(Final_Section)]);
    end
    
end

if averageanswr == 1
% creates average intensity image from selected sections
[Avg1,Avg2,Avg3]=AverageIntensity(param_file_name,AveragingSectionInfo,ChannelInfo);
else
    Avg1=1;
    Avg2=1;
    Avg3=1;
end



%imwrite(Avg1,'avg_1.tif');
%imwrite(Avg2,'avg_2,tif');
%imwrite(Avg3,'avg_3,tif');

zTile=0;
section=0;
current=0;
compfilecounter=0;
resetcheck=0;

secs = (Final_Section - Initial_Section)+1;
total = (((param.mrows*param.mcolumns)*param.layers)*secs*3);

for i=Initial_Section:Final_Section
    
    %directory_name = [directory,'/',param.SampleID,'-',leadingzeros,num2str(i)];
    %imagename = [directory_name,'/'];
    
    %files=dir([imagename,'*_01.tif']);
    files = find_all_files(param, i);
  
      
      


    if (length(files)==param.mrows*param.mcolumns*param.layers) || param.mrows*param.mcolumns*2*param.layers || param.mrows*param.mcolumns*3*param.layers
        
        for ch=Initial_Channel:Final_Channel
            
            filecounter=0;
            
            if ch==1
                Avg=Avg1;
            elseif ch==2
                Avg=Avg2;
            elseif ch==3
                Avg=Avg3;
            end
            
            
            
            for z=1:param.layers
                zTile=section*param.layers+z;
                
                for y=1:param.mcolumns
                    yTile = (param.mcolumns - y) + 1;
                    
                    for x=1:param.mrows
                        
                        % This makes sure that the while loop gets 
                        % called at least once to pick up the current 
                        % file.  
                        channel=0;
                        
                        while channel~=ch && filecounter<length(files)
                            filecounter=filecounter+1;
                            current_file = files{filecounter};
                            tiff_file=files{filecounter}.name;
                            
                            [pathstr, name] = fileparts(tiff_file);
                            channel_number = name ((strfind (name, tokeniser3) + 2):end);
                            channel = str2num(channel_number);
                        end
                        
                        if ch==channel

                            %I=imread([imagename tiff_file]);
                            I = tv_open(param, current_file);
                            
                            I=double(I);
                            if averageanswr == 1
                            I=I./Avg;
                            else
                            end
                            %New_I=rot90(I);
                            New_I=rot90(I((clipwidth(1)+1):(param.rows - (clipwidth(3))),(clipwidth(2)+1):(param.columns - (clipwidth(4)))));
                            
                            if mod(y,2)==1
                                xTile = (param.mrows - x) + 1;
                            else
                                xTile = x;
                            end
                            
                            if zTile<10
                                leadingzerosz = '00';
                            elseif (zTile>=10) && (zTile<100)
                                leadingzerosz = '0';
                            else
                                leadingzerosz = '';
                            end
                            
                            if yTile<10
                                leadingzerosy = '00';
                            elseif (yTile>=10) && (yTile<100)
                                leadingzerosy = '0';
                            else
                                leadingzerosy = '';
                            end
                            
                            if xTile<10
                                leadingzerosx = '00';
                            elseif (xTile>=10) && (xTile<100)
                                leadingzerosx = '0';
                            else
                                leadingzerosx = '';
                            end
                            
                            New_I16=uint16(New_I);
                            
                            if outputdir == 0
                                newimagefilename = [directory '/',param.SampleID,'-Mosaic/Ch',num2str(ch),'_Tile_Sections_',num2str(Initial_Section),'_to_',num2str(Final_Section),'/Tile_Z',leadingzerosz,num2str(zTile),'_Y',leadingzerosy,num2str(yTile),'_X',leadingzerosx,num2str(xTile),'.tif'];
                            else
                                newimagefilename = [outputdir '/',param.SampleID,'-Mosaic/Ch',num2str(ch),'_Tile_Sections_',num2str(Initial_Section),'_to_',num2str(Final_Section),'/Tile_Z',leadingzerosz,num2str(zTile),'_Y',leadingzerosy,num2str(yTile),'_X',leadingzerosx,num2str(xTile),'.tif'];
                            end
                            
                            if exist(newimagefilename,'file') == 2
                            else
                                imwrite(New_I16,newimagefilename,'tiff','Compression','none');
                            end
                            sprintf('Writing image: Tile_Z%s%d_Y%s%d_X%s%d.tif\n',leadingzerosz,zTile,leadingzerosy,yTile,leadingzerosx,xTile);
                            current = current+1;
                            clc;
                            percent = round((current/total)*100);
                            sprintf('Saving image %d out of %d. Operation %d%% complete.',current,total,percent)
                        end
                    end
                end
            end
        end
    end
    section=section+1;
    str = ['Section ', num2str(i), ' Complete'];
    disp(str)
end
str = ['Export Complete'];
disp(str)

