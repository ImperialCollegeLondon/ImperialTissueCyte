function base_name = find_file_name_base( params )
% This returns the base file name that each tile uses. 
% This is a time stamp of when the mosaic began.
% 

% This is in the format like: 8/21/2015 2:48:42 PM


date = params.acqDate;
l = length(params.acqDate);

day_start = 0;
year_start = 0;

for i = 1:l
    if(date(i) == '/')
        month_part = date(1:i-1);
        day_start = i+1;
        break;
    end
end


for i=day_start:l
   if(date(i) == '/')
       day_part =  date(day_start:(i-1));
       year_start = i+1;
   end
end

year_part = date(year_start:year_start+3);

space_index = strfind(date, ' ');
space_index = space_index(1);

time_string = date(space_index+1:end);
colon_index = strfind(time_string, ':');

hour_part = time_string(1:colon_index(1)-1);
minute_part = time_string(colon_index(1)+1:colon_index(2)-1);


if(time_string(end) == ' ')
    am_pm = time_string(end-2:end-1);
else
    am_pm = time_string(end-1:end);
end



if(am_pm(1) == 'P')
    if hour_part == '12'
        % When it's say 12:15 PM, we want to return 1215, not 2415
        hour_part = 12;
    else
        hour_part = str2num(hour_part) + 12;
    end
elseif (am_pm(1) == 'A')
    if hour_part == '12'
        % When it's say 12:15 AM, we want to return 0015, not 2415
        hour_part = 0;
    else
        hour_part = str2num(hour_part) ;
    end
    
else
    error('Error parsing mosaic file.  Please contact TissueVision');
end

base_name = sprintf('%02d%02d%02d-%02d%02d', str2num(month_part), str2num(day_part),str2num(year_part), hour_part, str2num(minute_part));

end

