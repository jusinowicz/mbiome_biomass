
Soil microbiome and above-ground biomass analysis. Meta analysis only. See 
Averil et al. 2022 for motivation and description of the (original) database. 

Workflow implemented here: data_main.R
 1.Import the database files, clean, join, check against Averil papers  
 2.Filter to biomass
 3.Assign controls and calculate response ratios.
   e.g. log (biomass_treatment/biomass_sterile_soil)
 4.Use dredge in MuMIn to explore the space of models from covariates in the 
   the original data set.
 5.Import geospatial environmental data and join to original data based on lat,lon: 
    SoilBase soil characteristics, WorldClim annual mean climate descriptors.
 6.Fit RandomForests models with the environmental covariates. Use a boostrap
   approach to generate distriubtions of variable importance.
 7.Based on the top e.g. 14 covariates from RF variable importance, do another
   round of dredge to explore the top environmental covariaties. 

Additional files and directory structure: 
data_prelim.R: Preliminary workflow and investigation before the code was cleaned and organized into data_main.R
functions/useful.R: custom functions that are useful for doing some of the processing, moved here to make the main files cleaner
data: This directory contains the core data files for the analysis. It also includes prelimiary figures and other results
