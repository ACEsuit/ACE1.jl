

"""
`read_potential(fname::AbstractString; format = :auto)` : convenience
function to load an ACE1 potential. If `format == :auto` it tries to 
guess the format from the ending of the filename. If this fails, then 
one can try to pass the format explicitly as a kwarg, 
* `format = :json` for a raw json file 
* `format = :zip` for a zipped json file that was written using `JuLIP.FIO.zip_dict`
"""
function read_potential(fname::AbstractString; format = :auto)
   if format == :auto 
      if fname[end-3:end] == "json"
         format = :json
      elseif fname[end-2:end] == "zip"
         format = :zip
      else
         error("Unknown filename ending")
      end
   end 
   
   if format == :json
      D = load_dict(fname)
   elseif format == :zip
      D = unzip_dict(fname)
   else
      error("Unknown filename ending; maybe try to explicitly pass the format")
      return nothing 
   end
   
end