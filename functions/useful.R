#=============================================================================
#Just some useful functions to help with data cleaning and analyses.
#=============================================================================
# Function to extract and convert coordinates
extract_coordinates = function(location) {
  # Pattern for degree-minute-second format
  # The seconds are optional
  pattern_dms1 ="(\\d+)°(\\d+)'(\\d*)(''|'')?" #"(\\d+)°(\\d+)'(\\d+)(''|'')?" #
  #Second pass, once the lat/lon are pulled out, to break out each element
  pattern_dms2 = "\\d+\\.?\\d*|\\d+" # "(\\d+)°(\\d+)'(\\d+)''"
  # Pattern for decimal degree format
  pattern_dd = "(\\d+\\.\\d+)"

  # Extract latitude and longitude using regular expressions
  lat_lon_dms1 = unlist(regmatches(location, gregexpr(pattern_dms1, location)))
  ll_temp = paste(lat_lon_dms1,collapse="") 
  lat_lon_dms = unlist(regmatches(ll_temp, gregexpr(pattern_dms2, ll_temp)))
  lat_lon_dd = unlist(regmatches(location, gregexpr(pattern_dd, location)))

  # Convert degree-minute-second format to decimal degrees
  if(length(lat_lon_dms)>=6) { 
    lat_dms = as.numeric(lat_lon_dms[1]) + as.numeric(lat_lon_dms[2])/60 + as.numeric(lat_lon_dms[3])/3600
    lon_dms = as.numeric(lat_lon_dms[4]) + as.numeric(lat_lon_dms[5])/60 + as.numeric(lat_lon_dms[6])/3600
  }else{ 
    lat_dms = as.numeric(lat_lon_dms[1]) + as.numeric(lat_lon_dms[2])/60 
    lon_dms = as.numeric(lat_lon_dms[3]) + as.numeric(lat_lon_dms[4])/60 
  }

  # Determine if latitude is North or South and longitude is East or West
  if (grepl("S", location)) lat_lon_dms = -lat_dms
  if (grepl("W", location)) lat_lon_dms = -lon_dms

  # Convert decimal degree format to decimal degrees
  lat_dd = as.numeric(lat_lon_dd[1])
  lon_dd = as.numeric(lat_lon_dd[2])

  # Determine if latitude is North or South and longitude is East or West
  if (grepl("S", location)) lat_lon_dd = -lat_dd
  if (grepl("W", location)) lat_lon_dd = -lon_dd

    # Determine latitude and longitude based on which format is present
  if (!is.na(lat_dms)) {
    lat = lat_dms
    lon = lon_dms
  } else if (!is.na(lat_dd)) {
    lat = lat_dd
    lon = lon_dd
  } else {
    lat = lon = NA
  }

  # Return latitude and longitude
  return(c(lat, lon))

}