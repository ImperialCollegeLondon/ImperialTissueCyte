function image = tv_open( params, path )
    % Opens a file that is given by the struct created by the 
    % find_all_files functio.
    % 
    % If the file was not found, this generates a blank tile. 
    
    
    if(path.found_it)
        image = imread(path.name);
    else
        %str = sprintf('missing: %s', path.name)
        image = zeros(params.rows);
    end
    
    
end

