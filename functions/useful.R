#=============================================================================
#Just some useful functions to help with data cleaning and analyses.
#=============================================================================
#Libraries
library(soilDB) #soilgrids data
# library(rnoaa) #average climate data 
# options(noaakey = "your_noaa_token")
library(geodata) #average climate data from WorldClim
library(sf)
library(httr)
library(jsonlite)
library(raster)

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

#=============================================================================
#Wrap the APIs for the environmental data sets in functions that recursively
#access the desired lat lon. 
#=============================================================================

####Soil properties per location
#SoilGrids18 0-15 cm depths: https://rest.soilgrids.org/query?lon=%f&lat=%f&depth=0-15cm
#All the mean quantities: c("bdodmean","cecmean","cfvomean","claymean",
#                            "nitrogenmean","phh2omean","sandmean", "siltmean",
#                           "socmean","ocdmean","wv0033mean", "wv1500mean" )
#url1="https://rest.isric.org/soilgrids/v2.0/properties/query?lon=173&lat=44"
#url <- sprintf("https://rest.isric.org/soilgrids/v2.0/properties/query?lon=%f&lat=%f&depth=0-15cm", lon, lat)
# response <- GET(url)
# data <- fromJSON(content(response, "text"))

get_SoilGrids = function (locations) {

  #Number of study sites
  nlocs = dim(locations)[1]

  #Variables to get
  variables_get = c("bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", 
                "sand", "silt", "soc", "ocd", "ocs",
                "wv0033", "wv1500" )

  variables = c("bdodmean","cecmean","cfvomean","claymean",
                           "nitrogenmean","phh2omean","sandmean", "siltmean",
                          "socmean","ocdmean","wv0033mean", "wv1500mean" )
  nvars = length(variables)

  #Initialize the final data frame
  df_return = as.data.frame(matrix(0,nlocs,nvars))
  colnames(df_return) = variables

  for (l in 1:nlocs){

    if(!is.na(locations[l,2])){
      #API request
      data_tmp=fetchSoilGrids( locations[l,], depth_interval = c("0-5", "5-15"), 
                variables = variables_get ) 

      #Add variables of interest into the final data frame
      for(v in 1:nvars){
        #Note: This is averaging over the two soil horizons for now
        df_return[l,variables[v]] = mean(data_tmp[[variables[v]]],na.rm=T) 
      }

    }else{}
  } 

  df_return = data.frame(studyGroup = locations$id, df_return)
  return (df_return)

}

####Averaged climate properties per location
# NOAA Token: EOWXxIevblWhQHvHeYCWptZexDWVZMRd
# Using WorldClim, via raster::getData
# This is written as two functions: one to grab and convert WorldClim data
# and a wrapper on that to do it iteratively across sites. 

# Function to fetch WorldClim data
fetch_worldclim_data = function(lat, lon, var = "bio", res = 10, path = tempdir()) {
  
  coords = data.frame(y=lat, x= lon)

  # Fetch the appropriate tile
  climate_tile = worldclim_tile(var = var, res = res, lon = lon, lat = lat, path = path)
  
  #Set the points to extract
  points = vect(coords,
               geom=c("x", "y"),
               crs = "EPSG:4326")

  # Extract data for the specified location
  climate_data = extract(climate_tile, points)
  
  return(climate_data)
}

get_climate = function (locations){ 
  
  # Initialize an empty data frame to store results
  results = data.frame()

  # Iterate over each location and fetch climate data
  for (i in 1:nrow(locations)) {
    lat = locations$lat[i]
    lon = locations$lon[i]
    location_id = locations$id[i]
    
    # Fetch WorldClim data for the current location
    climate_data = fetch_worldclim_data(lat, lon)
    
    # Convert to data frame and add location ID
    climate_df = as.data.frame(climate_data)
    colnames(climate_df) = c("ID", paste("bio", 1:19, sep="") ) 
    climate_df$location_id = location_id
    
    # Append the climate data to the results data frame
    results = rbind(results, climate_df)
  }


  return(results)

}