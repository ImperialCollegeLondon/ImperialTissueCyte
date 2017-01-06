function [Avg1,Avg2,Avg3] =AverageIntensity(param_file_name,AveragingSectionInfo,ChannelInfo)
[directory,name,extension]=fileparts(param_file_name);

param_file_ptr = fopen (param_file_name);
tokeniser = ':';
tokeniser2 =' ';
tokeniser3 ='_';
tokeniser4 ='-';

Initial_Averaging=AveragingSectionInfo(1);
Final_Averaging=AveragingSectionInfo(2);
Sections =Initial_Averaging:Final_Averaging;
Initial_Channel=ChannelInfo(1);
Final_Channel=ChannelInfo(2);
current=0;

param = struct('fileName', param_file_name);



n = 0;
line = 0;
while ((~(line == -1)) & (n < 38))
    line = fgetl (param_file_ptr);
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

im1=zeros(param.rows,param.columns);
im2=zeros(param.rows,param.columns);
im3=zeros(param.rows,param.columns);
ch1=0;
ch2=0;
ch3=0;



if param.Zscan==0
    param.layers=1;
end
secs = length(Sections);
total = (((param.mrows*param.mcolumns)*param.layers)*secs*Final_Channel);

for i=1:length(Sections)
    if Sections(i)<10
        leadingzeros = '000';
        
    else if (Sections(i)>=10) && (Sections(i)<100)
            leadingzeros = '00';
            
        else if (Sections(i)>=100) && (Sections(i)<1000)
                leadingzeros = '0';
                
            else
                leadingzeros= '';
                
            end
        end
    end
    
    imagename = [directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/'];
    files=dir([imagename,'*.tif']);
    
    if length(files)==param.mrows*param.mcolumns*param.layers || param.mrows*param.mcolumns*2*param.layers || param.mrows*param.mcolumns*3*param.layers
        
        for y=1:length(files)
            
            tiff_file=files(y).name;
            % rename file with 0 padding
            digit=14;
            [pathstr, name] = fileparts(tiff_file);
            image_date = name (1:(strfind (name, tokeniser4)-1));
            
            current = current+1;
            if current > total
                %keyboard
            end
            percent = round((current/total)*100);
            clc;
            sprintf('Calculating image %d out of %d. Operation %d%% complete.',current,total,percent)
            
            tokeniser_check = name(digit);
            while tokeniser_check ~= '-'
                digit=digit-1;
                tokeniser_check = name (digit);
            end
            file_number = name (strfind (name, tokeniser4)+1:(digit-1));
            image_number = name ((digit+1):(strfind (name, tokeniser3)-1));
            channel_number = name ((strfind (name, tokeniser3) + 2):end);
            channel = str2num(channel_number);
            number = str2num(image_number);
            character_number=length(name);
            
            if channel==1  && Initial_Channel==1
                I1=imread([imagename tiff_file]);
                I1=double(I1);
                im1=im1+I1;
                ch1=1;
                
                %sprintf('&s//%s-%s%s//%s',directory,param.SampleID,leadingzeros,num2str(i),tiff_file)
                
                
                if number<10 && character_number~=digit+8
                    %movefile([directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',tiff_file],[directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',image_date,'-',file_number,'-0000',image_number,'_01.tif']);
                elseif (number>=10) && (number<100) && character_number~=digit+8
                    %movefile([directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',tiff_file],[directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',image_date,'-',file_number,'-000',image_number,'_01.tif']);
                elseif (number>=100) && (number<1000) && character_number~=digit+8
                    %movefile([directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',tiff_file],[directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',image_date,'-',file_number,'-00',image_number,'_01.tif']);
                elseif (number>=1000) && (number<10000) && character_number~=digit+8
                    %movefile([directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',tiff_file],[directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',image_date,'-',file_number,'-0',image_number,'_01.tif']);
                end
                
            elseif channel==2 && (Initial_Channel==2 || Final_Channel-Initial_Channel>0)
                %sprintf('&s//%s-%s%s//%s',directory,param.SampleID,leadingzeros,num2str(i),tiff_file);
                I2=imread([imagename tiff_file]);
                I2=double(I2);
                im2=im2+I2;
                ch2=1;
                
                if number<10 && character_number~=digit+8
                    %movefile([directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',tiff_file],[directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',image_date,'-',file_number,'-0000',image_number,'_02.tif']);
                elseif (number>=10) && (number<100) && character_number~=digit+8
                    %movefile([directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',tiff_file],[directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',image_date,'-',file_number,'-000',image_number,'_02.tif']);
                elseif (number>=100) && (number<1000) && character_number~=digit+8
                    %movefile([directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',tiff_file],[directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',image_date,'-',file_number,'-00',image_number,'_02.tif']);
                elseif (number>=1000) && (number<10000) && character_number~=digit+8
                    %movefile([directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',tiff_file],[directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',image_date,'-',file_number,'-0',image_number,'_02.tif']);
                end
                
            elseif channel==3 && Final_Channel==3
                I3=imread([imagename tiff_file]);
                I3=double(I3);
                im3=im3+I3;
                ch3=1;
                
                if number<10 && character_number~=digit+8
                    %movefile([directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',tiff_file],[directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',image_date,'-',file_number,'-0000',image_number,'_03.tif']);
                elseif (number>=10) && (number<100) && character_number~=digit+8
                    %movefile([directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',tiff_file],[directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',image_date,'-',file_number,'-000',image_number,'_03.tif']);
                elseif (number>=100) && (number<1000) && character_number~=digit+8
                    %movefile([directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',tiff_file],[directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',image_date,'-',file_number,'-00',image_number,'_03.tif']);
                elseif (number>=1000) && (number<10000) && character_number~=digit+8
                    %movefile([directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',tiff_file],[directory,'/',param.SampleID,'-',leadingzeros,num2str(Sections(i)),'/',image_date,'-',file_number,'-0',image_number,'_03.tif']);
                end
                
            end
        end
    end
end


Avg1=im1./(param.mrows*param.mcolumns*length(Sections)*param.layers);
max1=max(max(Avg1));
Avg1=Avg1./max1;

Avg2 =im2./(param.mrows*param.mcolumns*length(Sections)*param.layers);
max2=max(max(Avg2));
Avg2=Avg2./max2;

Avg3 =im3./(param.mrows*param.mcolumns*length(Sections)*param.layers);
max3=max(max(Avg3));
Avg3=Avg3./max3;

str = ['Averaging Complete'];
disp(str)

