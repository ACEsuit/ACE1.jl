

export load_potential, save_potential

function _auto_format(fname)
   if fname[end-3:end] == "json"
      format = :json
   elseif fname[end-2:end] == "zip"
      format = :zip
   else
      error("Unknown filename ending; try to pass the format explicitly as a kwarg")
   end
   return format 
end

"""
`load_potential(fname::AbstractString; format = :auto)` : convenience
function to load an ACE1 potential. If `format == :auto` it tries to 
guess the format from the ending of the filename. If this fails, then 
one can try to pass the format explicitly as a kwarg, 
* `format = :json` for a raw json file 
* `format = :zip` for a zipped json file that was written using `JuLIP.FIO.zip_dict`
"""
function load_potential(fname::AbstractString; format = :auto)
   if format == :auto 
      format = _auto_format(fname)
   end 
   
   if format == :json
      D = load_dict(fname)
   elseif format == :zip
      D = unzip_dict(fname)
   else
      error("Unknown file format")
      return nothing 
   end
   
   return read_dict(D)
end

"""
`save_potential(fname::AbstractString, V; format = :auto)` : convenience
function to save an ACE1 potential to disk. If `format == :auto` it tries to 
guess the format from the ending of the filename. If this fails, then 
one can try to pass the format explicitly as a kwarg, 
* `format = :json` for a raw json file 
* `format = :zip` for to write a compressed json file using `JuLIP.FIO.zip_dict`
"""
function save_potential(fname::AbstractString, V; format = :auto)
   if format == :auto 
      format = _auto_format(fname)
   end 

   D = write_dict(V)
   if format == :json 
      JuLIP.FIO.save_dict(fname, D)
   elseif format == :zip 
      JuLIP.FIO.zip_dict(fname, D)
   else      
      error("Unknown file format")
   end
end

