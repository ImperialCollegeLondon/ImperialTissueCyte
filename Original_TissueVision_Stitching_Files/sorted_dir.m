function  sorted_files  = sorted_dir( path )


unsorted_files = dir(path);

files_map = containers.Map('KeyType', 'int32', 'ValueType', 'any');

number_files = length(unsorted_files);

files_indices = zeros(number_files);

sorted_files = unsorted_files;

for i = 1:number_files
    f = unsorted_files(i);
    file_number = find_file_number(f.name);
    files_map(file_number) = f;
    files_indices(i) = file_number;
end

sorted_indices = sort(files_indices);

for i = 1:number_files
    sorted_files(i) = files_map(sorted_indices(i));
end


end


function file_number = find_file_number(path)



str_end = length(path);
index = str_end;

while( (path(index) ~= '_') && (index > 0))
    index = index - 1;
end

end_number = index-1;

while( (path(index) ~= '-') && (index > 0))
    index = index -1;
end

start_number = index + 1;

number_text = path(start_number:end_number);
file_number = int32(str2num(number_text));

end