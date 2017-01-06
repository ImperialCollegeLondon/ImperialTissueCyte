function missing_tiles = find_missing_tiles(base_directory, file_name, channel_number, first_index, last_index )
% This returns a list of file names of tiles that should be in a directory 
% but were not found. 
%
% This function does not try to fix the missing tiles. 

missing_tiles = {};

for i = first_index:last_index 
    full_path = sprintf('%s/%s-%d_%02d.tif', base_directory, file_name, i, channel_number)
    if(exist(full_path)~=2)
       % The file does not exist
       display(sprintf('Could not find: %s\n', full_path));
       missing_tiles{end+1} = full_path;
       
       keyboard
    end
end



end

