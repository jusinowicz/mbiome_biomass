#=============================================================================
# Soil microbiome and above-ground biomass analysis
#
# This project combines three different data sets to investigate how
# environmental drivers interact with soil microbiota to impact aboveground 
# biomass.
#
# The three datasets are: 
#	1. Meta analysis data set from Averill et al. 2022 10.1038/s41564-022-01228-3
# 	2. 
#	3. 
#
#=============================================================================
# load libraries
#=============================================================================
library(dplyr)
library(tidyverse)
library(metafor)

#library(rgee)
source("./functions/useful.R")
#=============================================================================
# load data sets, do some initial cleans and sanity checks. 
#=============================================================================
#Averill 2022 meta analysis data. Two separate files, representing two 
#different collection stages and potentially different methods? 
dfa1 = read.csv(file="./data/averil_meta1.csv")
dfa2 = read.csv(file="./data/averil_meta2.csv")
use_studies = read.csv(file = "./data/averil_sites.csv")

#####Some cleans###############################################################
#Reads as character because of "?"
dfa1$IsLatestTimePoint = as.numeric(dfa1$IsLatestTimePoint )
dfa2$IsLatestTimePoint = as.numeric(dfa2$IsLatestTimePoint )

#Join all the rows
dfa = full_join(dfa1,dfa2)

#Remove these rows (non-entries? )
dfa = dfa[-220,]
dfa = dfa[-673,]

# Find the coordinates for each site, pull them out, convert all to 
# degree decimal format, add them as new numerical columns. 
coords_temp = t(apply(as.matrix(dfa$location), 1, extract_coordinates))
dfa$lat = coords_temp[,1]
dfa$lon = coords_temp[,2]

#Do some standardizing of variables on the inocType
dfa$inocType = gsub("sterile", "ster", dfa$inocType) 
unique(dfa$inocType) 

#Add a column for study 
study_key = data.frame(DOI=unique(dfa$DOI), Study=1:length(unique(dfa$DOI)))
dfa= left_join(dfa, study_key, by = "DOI")

#Remove beginning time-points from studies with muliple time points: 
dfa = dfa[(!is.na(dfa$comparisonGroup)),]

#Add a column to combine experiment (these are grouped within)
dfa$studyGroup = NA
dfa$studyGroup = paste(dfa$Study,dfa$comparisonGroup,sep=".")
###############################################################################

####Some checks on data set####################################################
#Check dfa size. dim[2] should be bigger than dim(dfa1)[2] and dim(dfa2)[2]
#and also these should all be equal: 
dim(dfa)[2]
dim(dfa2)[2] + length (colnames(dfa1[!(colnames(dfa1) %in% colnames(dfa2))] ))
dim(dfa1)[2] + length (colnames(dfa2[!(colnames(dfa2) %in% colnames(dfa1))] ))

#Another check: these are the studies used in Averill et al. 2022
#Convert all characters in the DOIs to uppercase for matching
dfa$DOI = toupper(dfa$DOI)
use_studies$DOI = toupper(use_studies$DOI)
dfa_av = dfa[dfa$DOI %in% unique(use_studies$DOI), ]
###############################################################################

#=============================================================================
# Filter the dataset. 
#	Averill et al. uses 3 big filters: 
#	1. Single-source inoculum 
#	2. Absence of non-inoculated control
#	3. Response doesn't include biomass
#
# Create other datasets for other response variables. How many are there? 13 
# so far
#=============================================================================
#Filter out single-source and other unwanted inocula treatments: 
# c("funal hyphae", "roots", "spores", "mycorrhizal roots",
# "chopped roots","infected roots") 
# mycorrhizal roots and chopped roots are in original study, but not sure 
# how useful they'll be here...

in_filter = c( "fungal hyphae", "roots", "infected roots")  
dfa_in = dfa %>%
  group_by(DOI) %>%
  filter(!any(inocType %in% in_filter)) %>% as.data.frame()

# Filter out experiments without a control, where control is either
# "no" or "ster"
con_filter = c("no", "ster")  
dfa_con = dfa_in %>%
  group_by(DOI) %>%
  filter(any(inocType %in% con_filter)) %>% as.data.frame()

#Break data into sets based on the response. Possible responses are:
# 1. TotBiomass: N = 360, studies = 17
dfa_TB = subset(dfa_con, !is.na(dfa_con$TotBiomass) )
dim(dfa_TB)[1]
length(unique(dfa_TB$DOI)) 

# 2. AboveBio: N = 314, studies = 21
dfa_AB = subset(dfa_con, !is.na(dfa_con$AboveBio) )
dim(dfa_AB)[1]
length(unique(dfa_AB$DOI)) 

# 3. BelowBio: N = 123, studies = 12
dfa_BB = subset(dfa_con, !is.na(dfa_con$BelowBio) )
dim(dfa_BB)[1]
length(unique(dfa_BB$DOI)) 

# 3B. Combine AboveGround and Total: N = 613, studies = 36
dfa_B = subset(dfa_con, !is.na(dfa_con$AboveBio) | !is.na(dfa_con$TotBiomass)  )
dim(dfa_B)[1]
length(unique(dfa_B$DOI)) 

#Calculate TotBiomass when it hasn't been calculated yet. 
dfa_B$TotBiomass[ is.na(dfa_B$TotBiomass)] = 
                  rowSums( dfa_B[is.na(dfa_B$TotBiomass),][,c(27,29)],na.rm=T)
dfa_B$TotBioError[ is.na(dfa_B$TotBioError)] = 
                  rowSums( dfa_B[is.na(dfa_B$TotBioError),][,c(28,30)],na.rm=T)


# 4. Yield N = 0
dfa_Y = subset(dfa_con, !is.na(dfa_con$Yield) )
dim(dfa_Y)[1]
length(unique(dfa_Y$DOI)) 

# 5. height: N = 164, studies = 16
dfa_h = subset(dfa_con, !is.na(dfa_con$height) )
dim(dfa_h)[1]
length(unique(dfa_h$DOI)) 

# 6. CoverPerc: N = 0 
dfa_CP = subset(dfa_con, !is.na(dfa_con$CP) )
dim(dfa_CP)[1]
length(unique(dfa_CP$DOI)) 

# 7. seedlingSurvPerc N = 115, studies = 13
dfa_sSP = subset(dfa_con, !is.na(dfa_con$seedlingSurvPerc) )
dim(dfa_sSP)[1]
length(unique(dfa_sSP$DOI)) 

# 8. LeafArea.cm.2.	N = 10, studies = 1
dfa_LA = subset(dfa_con, !is.na(dfa_con$LeafArea.cm.2.) )
dim(dfa_LA)[1]
length(unique(dfa_LA$DOI)) 

# 9. LeafPhosphorous.mg.kgDryWeight. N = 381, studies = 38
dfa_LP = subset(dfa_con, !is.na(dfa_con$LeafPhosphorous.mg.kgDryWeight.) )
dim(dfa_LP)[1]
length(unique(dfa_LP$DOI)) 

#10. BasalAreaGrowthSeedlings.mm.2. N = 15, studies = 1 
dfa_BAGS = subset(dfa_con, !is.na(dfa_con$BasalAreaGrowthSeedlings.mm.2.) )
dim(dfa_BAGS)[1]
length(unique(dfa_BAGS$DOI))

#11. Stem.diameter N = 12, studies = 1
dfa_Sd = subset(dfa_con, !is.na(dfa_con$Stem.diameter) )
dim(dfa_Sd)[1]
length(unique(dfa_Sd$DOI))

#12. Germination. N = 3, studies = 1
dfa_G = subset(dfa_con, !is.na(dfa_con$Germination.) )
dim(dfa_G)[1]
length(unique(dfa_G$DOI))

#13. NumberofGerminatedSeedlings N = 0, stuies = 0
dfa_ngs= subset(dfa_con, !is.na(dfa_con$NumberofGerminatedSeedlings) )
dim(dfa_ngs)[1]
length(unique(dfa_ngs$DOI))

#=============================================================================
# Use the controls and treatments to create response ratios
#=============================================================================
####
#Aboveground Biomass only 
#Create a new df with "control" columns for no and sterile soil:
a1 = dfa_AB %>% 
            filter (inocType %in% c("no") ) %>% 
            group_by(studyGroup) %>%
            mutate(inocType = paste0("con_", inocType)) %>%  # Rename Treatments for clarity
            pivot_wider(names_from = inocType, values_from =c(AboveBio, AboveBioError),
                         values_fill = NA) %>%
            data.frame()

a2 = dfa_AB %>% 
            filter (inocType %in% c("ster") ) %>% 
            group_by(studyGroup) %>%
            mutate(inocType = paste0("con_", inocType)) %>%  # Rename Treatments for clarity
            pivot_wider(names_from = inocType, values_from =c(AboveBio, AboveBioError), 
              values_fill = NA) %>%
            data.frame()

#Join these so there is a column for each control version
dfa_AB_con = left_join(dfa_AB, a1[,c(71:73)], by = "studyGroup" ) %>%
            left_join(a2[,c(71:73)],by = "studyGroup")

#Create the final Control and Control SD columns. If a study has both sterile and 
#no treatments, it takes the average of the two (and the average of the SDs). 
dfa_AB_con$Control = apply(dfa_AB_con[,c(75,77)], 1, mean, na.rm=T )   
dfa_AB_con$ControlErr = apply(dfa_AB_con[,c(76,78)], 1, function(x) sqrt(sum(x^2,na.rm=T)) )   

#Calculate the effect sizes 
es_dfa_AB = escalc(measure = "ROM", 
                  m1i = dfa_AB_con$AboveBio, 
                  m2i = dfa_AB_con$Control,
                  sd1i = dfa_AB_con$AboveBioError,
                  sd2i = dfa_AB_con$ControlErr, 
                  n1i = matrix(1,dim(dfa_AB_con)[1],1), 
                  n2i = matrix(1,dim(dfa_AB_con)[1],1))      
################
#Total Biomass
a1 = dfa_B %>% 
            filter (inocType %in% c("no") ) %>% 
            group_by(studyGroup) %>%
            mutate(inocType = paste0("con_", inocType)) %>%  # Rename Treatments for clarity
            pivot_wider(names_from = inocType, values_from =c(TotBiomass, TotBioError),
                         values_fill = NA) %>%
            data.frame()

a2 = dfa_B %>% 
            filter (inocType %in% c("ster") ) %>% 
            group_by(studyGroup) %>%
            mutate(inocType = paste0("con_", inocType)) %>%  # Rename Treatments for clarity
            pivot_wider(names_from = inocType, values_from =c(TotBiomass, TotBioError), 
              values_fill = NA) %>%
            data.frame()

#Join these so there is a column for each control version
dfa_B_con = left_join(dfa_B, a1[,c(71:73)], by = "studyGroup" ) %>%
            left_join(a2[,c(71:73)],by = "studyGroup")

#Create the final Control and Control SD columns. If a study has both sterile and 
#no treatments, it takes the average of the two (and the average of the SDs). 
dfa_B_con$Control = apply(dfa_B_con[,c(75,77)], 1, mean, na.rm=T )   
dfa_B_con$ControlErr = apply(dfa_B_con[,c(76,78)], 1, function(x) sqrt(sum(x^2,na.rm=T)) )   

#Calculate the effect sizes 
es_dfa_B = escalc(measure = "ROM", 
                  m1i = dfa_B_con$TotBiomass, 
                  m2i = dfa_B_con$Control,
                  sd1i = dfa_B_con$TotBioError,
                  sd2i = dfa_B_con$ControlErr, 
                  n1i = matrix(1,dim(dfa_B_con)[1],1), 
                  n2i = matrix(1,dim(dfa_B_con)[1],1))  

#Attach this to the main dataframe for the stats: 
dfa_B_con = cbind(dfa_B_con, es_dfa_B)   

#Now remove the rows that are for the controls: 
dfa_B_con2 = dfa_B_con[dfa_B_con$yi>0, ]  #create new data frame
dfa_B_con2 = dfa_B_con2[!(is.na(dfa_B_con2$Study)),]
#Two studies in dfa_B_con have both no and sterile treatments. How to handle?
#Remove one of the two for now. 
dfa_B_con2 = dfa_B_con2[dfa_B_con2$inocType != "no" & dfa_B_con2$inocType != "ster", ] 

#Remove studies without SD/SE? 
dfa_B_con2 = dfa_B_con2[ dfa_B_con2$vi>0, ]

#=============================================================================
# Start thinking about the random-effects models we can explore
#=============================================================================

# Basic (nullish), 2-level, studies independent: AIC = 472.4273
mod1 = rma(yi,vi, data= dfa_B_con2 )

# Model 1 with random effects of Study: AIC = 1126.5173
mod1_r1 = rma.mv(yi,vi, random = ~ 1 | Study, data= dfa_B_con2 )

# Model 1, three-level, studyGroup within Study: AIC = 357.7891 
mod1_r2 = rma.mv(yi,vi, random = ~ 1 | Study/studyGroup, data= dfa_B_con2 )

# Add in some treatment effects. Try a few of the obvious ones before we
# try an all-out model selection on AIC: 
# Model 2, three-level, inocType AIC = 351.21 
mo2_r2 = rma.mv(yi,vi, mods = ~ inocType,  random = ~ 1 | Study/studyGroup, data= dfa_B_con2 )

# Model 2, AIC = 276.0063
mo2_r3 = rma.mv(yi,vi, mods = ~ inocType+Ecosystem+EcoRegion,  random = ~ 1 | Study/studyGroup, data= dfa_B_con2 )

#=============================================================================
# Get spatial environmental covariates for each of the locations. These data 
# sets are: 
# 1. Soil data from SoilGrids (via soilDB library)
# 2. Average climate characteristics from 
#=============================================================================
locations_use = data.frame(id = dfa_B_con2$studyGroup ,lat=(dfa_B_con2$lat), lon=(dfa_B_con2$lon),stringsAsFactors = FALSE )
locations = locations_use[!is.na(locations_use[,2]),]

#Soil charactertistics
soil_use = get_SoilGrids(locations)
#save(file = "./data/soilgrids1.var", soil_use)
#load("./data/soilgrids1.var")

#Append these to the df
soil_use_key = soil_use[!duplicated(soil_use$studyGroup),]
dfa_B_con2 = left_join(dfa_B_con2, soil_use_key, by = "studyGroup" )

#Average climate variables
climate_use = get_climate(locations)

#=============================================================================
#Use Google Earth Engine to streamline getting the different satellie layers. 
#reticulate::conda_create(envname = "rgee_env", packages = "python=3.8")
#reticulate::use_condaenv("rgee_env", required = TRUE)
#ee_install_upgrade()
#=============================================================================
# ee_Initialize()

# #Get the locations
# locations_use = cbind((dfa_B_con2$lat), (dfa_B_con2$lon) )
# nlocs = dim(locations_use)[1]

# #Loop over the locations
# for (l in 1:nlocs){

#   point_ee=ee$Geometry$Point(c(locations_use[l,]))


# } 