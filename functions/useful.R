#=============================================================================
#Just some useful functions to help with data cleaning and analyses.
#=============================================================================
# Function to extract and convert coordinates
extract_coordinates = function(location) {
  # Extract latitude and longitude using regular expressions
  lat_lon = regmatches(location, gregexpr("\\d+\\.\\d+|\\d+", location))[[1]]
  
  # Convert degree-minute-second format to decimal degrees
  lat = as.numeric(lat_lon[1]) + as.numeric(lat_lon[2])/60 + as.numeric(lat_lon[3])/3600
  lon = as.numeric(lat_lon[4]) + as.numeric(lat_lon[5])/60 + as.numeric(lat_lon[6])/3600
  
  # Determine if latitude is North or South and longitude is East or West
  if (grepl("S", location)) lat = -lat
  if (grepl("W", location)) lon = -lon
  
  # Return latitude and longitude
  return(c(lat, lon))
}