#=============================================================================
# This file is the original work flow for cleaning and investigating the data
# sets. It is not the cleanest, most well-ordered version of the analysis. 
# Please see 
#
# Soil microbiome and above-ground biomass analysis
#
# This project combines three different data sets to investigate how
# environmental drivers interact with soil microbiota to impact aboveground 
# biomass.
#
# The three datasets are: 
#	1. Meta analysis data set from Averill et al. 2022 10.1038/s41564-022-01228-3
# 2. Three experimental innoculations on trees in Wales, Ireland, and Mexico 
#	3. Data from Funga.
#
# Note: As of the latest version of this code, only data set 1 is available
#=============================================================================
# load libraries
#=============================================================================
library(dplyr)
library(tidyverse)
#ME models
library(metafor)
library(MuMIn)
#PCA
library(FactoMineR)
#Machine learning
library(keras)
library(randomForest)

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

# 3B. Combine AboveGround and BelowBio: Total: N = 613, studies = 36
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
# 2. Average climate characteristics from WorldClim
#=============================================================================
locations_use = data.frame(id = dfa_B_con2$studyGroup ,lat=(dfa_B_con2$lat), lon=(dfa_B_con2$lon),stringsAsFactors = FALSE )
locations = locations_use[!is.na(locations_use[,2]),]

#Soil charactertistics
#soil_use = get_SoilGrids(locations)
#save(file = "./data/soilgrids1.var", soil_use)
load("./data/soilgrids1.var")

#Append these to the df
soil_use_key = soil_use[!duplicated(soil_use$studyGroup),]
dfa_B_con2 = left_join(dfa_B_con2, soil_use_key, by = "studyGroup" )

#Average climate variables
#climate_use = get_climate(locations)
#colnames(climate_use)[21] = "studyGroup"
#save(file = "./data/worldclim_bio.var", climate_use)
load("./data/worldclim_bio.var")

#Append these to the df
climate_use_key = climate_use[!duplicated(climate_use$studyGroup),]
dfa_B_con2 = left_join(dfa_B_con2, climate_use_key, by = "studyGroup" )
dfa_B_con2 = dfa_B_con2[,-95]

#=============================================================================
# PCA on the environmental covariates to assess colinearity
#=============================================================================
soil_pc = prcomp(na.omit(soil_use[,-1]),
             center = TRUE,
            scale. = TRUE)

summary(soil_pc)

climate_pc = prcomp(na.omit(climate_use[,-c(1,21)]),
             center = TRUE,
            scale. = TRUE)

summary(climate_pc)


#=============================================================================
# Now that we have these environmental covariates, explore some RandomForests 
# models to see which variables seem to drive patterns.
# Does it make more sense to do this on response ratio or biomass itself? 
# 
# Compare with and without categorical variables like Study and studyGroup to 
# see which studies are showing results that are worth looking ath more 
# closely. 
#=============================================================================
#Remove NAs
dfa_ml_use = dfa_B_con2[!is.na(dfa_B_con2$bdodmean), ]

#First grab the numeric values we want, including the response, and 
#standardize.
#81 is yi, the response ratio. 25 is TotBiomass. 
col_ml = c(81,83:113)
dfa_ml = dfa_ml_use[,col_ml ]
dfa_ml = as.data.frame(scale(dfa_ml))

colnames(dfa_ml)[14:32] = c("AMT", "MDR", "Iso","SeaT","MaxT","MinT","ART",
  "MTempW", "MTempD","MTempH","MTempC","APr","PrW","PrD","SeaPr","PrWQ",
  "PrDQ", "PrH","PrC") 

#One-hot encoding: Study, studyGroup, inocType, Ecosystem
col_cat = c(73,74,3,9)
#Pull out the categoricals
dfa_cat = dfa_ml_use[,col_cat]
col_l = length(col_cat)
dfa_cat_new = NULL #Store them here

for(c in 1:col_l){

  new_cat = dfa_cat[,c]
  #Use gsub to remove spaces, commas, and slashes otherwise RF
  #chokes and fails. 
  new_cat = gsub("[ ,/]", "", new_cat)
  new_cat = as.factor(new_cat) #Make them factors, then
  new_cat_tmp = model.matrix(~0+new_cat) #This is really effective
  #Rename columns
  colnames(new_cat_tmp) = paste(colnames(dfa_cat)[c],levels(new_cat),sep="")
  #Add to the data frame
  dfa_cat_new = cbind(dfa_cat_new, new_cat_tmp)

}

#####Fit two versions of the model: One without, then one with the 
#####the categoricals.
# 1. Continuous values only:
#Split data for training and testing: 
ind = sample(2, nrow(dfa_ml), replace = TRUE, prob = c(0.8, 0.2))
train_dfa_ml = dfa_ml [ind==1,]
test_dfa_ml = dfa_ml [ind==2,]

#Tuning the full RF model: 
t = tuneRF(train_dfa_ml[,-1], train_dfa_ml[,1],
   stepFactor = 0.5,
   plot = TRUE,
   ntreeTry = 150,
   trace = TRUE,
   improve = 0.05)

#Get mtry with the lowest OOB Error
# t[ as.numeric(t[,2]) < 0 ] = 1
mtry_use = as.numeric(t[which(t == min(t),arr.ind=T)[1],1])  

#Basic RF fitting
model_form = "yi ~."
biomass_rf = randomForest (as.formula(model_form),
  data=train_dfa_ml, proximity=TRUE, mtry = mtry_use)

#Prediction
pred_test_rf = predict(biomass_rf, test_dfa_ml)

#RMSE between predictions and actual
rmse_rf = sqrt( mean((pred_test_rf - test_dfa_ml[,1])^2,na.rm=T) )

#Look at variable importance: 
#fig.name = paste("varImpPlot3",".pdf",sep="")
#pdf(file=fig.name, height=8, width=8, onefile=TRUE, family='Helvetica', pointsize=16)

#####
#2. With the categorical variables
#Combine both data sets
dfa_ml = cbind(dfa_ml,dfa_cat_new)

train_dfa_ml = dfa_ml [ind==1,]
test_dfa_ml = dfa_ml [ind==2,]

#Tuning the full RF model: 
t2 = tuneRF(train_dfa_ml[,-1], train_dfa_ml[,1],
   stepFactor = 0.5,
   plot = TRUE,
   ntreeTry = 150,
   trace = TRUE,
   improve = 0.05)

#Get mtry with the lowest OOB Error
# t[ as.numeric(t[,2]) < 0 ] = 1
mtry_use = as.numeric(t2[which(t == min(t),arr.ind=T)[1],1])  

#Basic RF fitting
model_form = "yi ~."
biomass_rf_cat = randomForest (as.formula(model_form),
  data=train_dfa_ml, proximity=TRUE, mtry = mtry_use)

#Prediction
pred_test_rf_cat = predict(biomass_rf_cat, test_dfa_ml)

#RMSE between predictions and actual
rmse_rf_cat = sqrt( mean((pred_test_rf_Cat - test_dfa_ml[,1])^2,na.rm=T) )

#Look at variable importance: 
#fig.name = paste("varImpPlot3",".pdf",sep="")
#pdf(file=fig.name, height=8, width=8, onefile=TRUE, family='Helvetica', pointsize=16)


par(mfrow = c(1,2))

p1 = varImpPlot(biomass_rf,
           sort = T,
           n.var = 40,
           main = "Variable Importance"
)

p2 = varImpPlot(biomass_rf_cat,
           sort = T,
           n.var = 40,
           main = "Variable Importance"
)

# BIO1 = Annual Mean Temperature
# BIO2 = Mean Diurnal Range (Mean of monthly (max temp - min temp))
# BIO3 = Isothermality (BIO2/BIO7) (×100)
# BIO4 = Temperature Seasonality (standard deviation ×100)
# BIO5 = Max Temperature of Warmest Month
# BIO6 = Min Temperature of Coldest Month
# BIO7 = Temperature Annual Range (BIO5-BIO6)
# BIO8 = Mean Temperature of Wettest Quarter
# BIO9 = Mean Temperature of Driest Quarter
# BIO10 = Mean Temperature of Warmest Quarter
# BIO11 = Mean Temperature of Coldest Quarter
# BIO12 = Annual Precipitation
# BIO13 = Precipitation of Wettest Month
# BIO14 = Precipitation of Driest Month
# BIO15 = Precipitation Seasonality (Coefficient of Variation)
# BIO16 = Precipitation of Wettest Quarter
# BIO17 = Precipitation of Driest Quarter
# BIO18 = Precipitation of Warmest Quarter
# BIO19 = Precipitation of Coldest Quarter


#=============================================================================
# Use dredge to explore the ME model space. 
#=============================================================================
#Remove NAs
dfa_ml_use = dfa_B_con2[!is.na(dfa_B_con2$bdodmean), ]
dfa_ME = dfa_ml_use 

colnames(dfa_ME)[95:113] = c("AMT", "MDR", "Iso","SeaT","MaxT","MinT","ART",
  "MTempW", "MTempD","MTempH","MTempC","APr","PrW","PrD","SeaPr","PrWQ",
  "PrDQ", "PrH","PrC") 


#Model with the top 14 covariates from the RandomForest variable importance
model_full = rma.mv(yi,vi, mods = ~bdodmean+wv0033mean+AMT+PrC+SeaPr+cecmean+
                                    PrW+Iso+PrWQ+MDR+ocdmean+siltmean+
                                    nitrogenmean+APr,
                                    random =   ~ 1 | Study/studyGroup, 
                                    data= dfa_ME, method = "ML")

options(na.action = "na.fail") #Needed to run dredge
eval(metafor:::.MuMIn) #Need this to tell MuMIn how to dredge correctly with metafor
model_dredge = dredge(model_full, trace=2)
#save(file="./data/dredged_RE_model.var", model_dredge)
options(na.action = "na.omit") # set back to default
 
#Examine models With AICc value no more than 2 units away from the best model
best_mods = subset(model_dredge, delta <= 2, recalc.weights=FALSE)

#Multi-model inference
avg_model = summary(model.avg(best_mods))

#Top model
top_model = get.models(model_dredge,subset = 1)[[1]]

# train_dfa_ml = dfa_ml [ind==1,]
# test_dfa_ml = dfa_ml [ind==2,]
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