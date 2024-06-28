#=============================================================================
#
# Soil microbiome and above-ground biomass analysis. Meta analysis only. See 
# Averil et al. 2022 for motivation and description of the (original) database. 
# 
# Workflow implemented here: 
# 1.Import the database files, clean, join, check against Averil papers  
# 2.Filter to biomass
# 3.Assign controls and calculate response ratios.
#   e.g. log (biomass_treatment/biomass_sterile_soil)
# 4.Use dredge in MuMIn to explore the space of models from covariates in the 
#   the original data set.
# 5.Import geospatial environmental data and join to original data based on lat,lon: 
#    SoilBase soil characteristics, WorldClim annual mean climate descriptors.
# 6.Fit RandomForests models with the environmental covariates. Use a boostrap
#   approach to generate distriubtions of variable importance.
# 7.Based on the top e.g. 14 covariates from RF variable importance, do another
#   round of dredge to explore the top environmental covariaties.   
#=============================================================================
# load libraries
#=============================================================================
library(dplyr)
library(tidyverse)
library(gridExtra)
#ME models
library(metafor)
library(MuMIn)
#PCA
library(FactoMineR)
#Machine learning
library(keras)
library(randomForest)

#Custom functions:
source("./functions/useful.R")
#=============================================================================
# 1.Import the database files, clean, join, check against Averil papers 
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
# 2.Filter dataset to biomass
#	Averill et al. uses 3 big filters: 
#	1. Single-source inoculum 
#	2. Absence of non-inoculated control
#	3. Response doesn't include biomass
#
# See data_prelim.R for other 13 response variables.
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

#Break data into sets based on the response = total biomass
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


#=============================================================================
# 3.Assign controls and calculate response ratios.
#   e.g. log (biomass_treatment/biomass_sterile_soil)
#=============================================================================
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

##################################
#Calculate the effect sizes 
##################################
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

#output1 = dfa_B_con2[,colnames(dfa_B_con2)[c(1:5,7,25,71,72,73,74,81)] ]
#write.csv(file = "./data/bmbiome_studies.csv", output1)
#=============================================================================
# 4.Use dredge in MuMIn to explore the space of models from covariates in the 
#   the original data set.
#
# This has probably already been run and fitted, so look for the load option
# instead of running from scratch. Comment/uncomment as needed to run dredge
#=============================================================================
#Do some dredging across multiple variables: 
#Model with the top 14 covariates from the RandomForest variable importance
#Note: dropped FungalType: not significant, NAs reduce data 
mf1 = rma.mv(yi,vi, mods = ~ inocType+Ecosystem+EcoRegion+FieldGreenhouse,
                                    random =   ~ 1 | Study/studyGroup, 
                                    data= dfa_B_con2, method = "ML")

options(na.action = "na.fail") #Needed to run dredge
eval(metafor:::.MuMIn) #Need this to tell MuMIn how to dredge correctly with metafor

#####>>>>>>>>>>>>>>>Comment/uncomment here as needed <<<<<<<<<<<<<<<<<<<<
#mf1_dredge = dredge(mf1, trace=2)
#save(file="./data/dredged_RE_mf1.var", mf1_dredge)
load(file="./data/dredged_RE_mf1.var")
#####>>>>>>>>>>>>>>>Comment/uncomment here as needed <<<<<<<<<<<<<<<<<<<<

options(na.action = "na.omit") # set back to default

#Examine models With AICc value no more than 2 units away from the best model
best_mf1 = subset(mf1_dredge, delta <= 2, recalc.weights=FALSE)

#Multi-model inference
avg_mf1 = summary(model.avg(best_mf1))

#Top model
top_mf1 = get.models(mf1_dredge,subset = 1)[[1]]

#####To export summaries into csv: 
# fp = "./data/best_re_mf1.csv"                                                                                                
# sink(file = fp)                                                                                                              
# summary(top_mf1)                                                                                                           
# sink()    

#Plot the data by Ecosystem. Look at the 3 that are significant: dry perennial grassland,
#shrubland/forest, tropical dry forest. These all have only one study, one site, and 
#above-average biomass response ratio.  
#fig.name = paste("./data/biomass_response1",".pdf",sep="")
#pdf(file=fig.name, height=8, width=8, onefile=TRUE, family='Helvetica', pointsize=16)
ggplot(data=dfa_B_con2, aes(x = factor(Study), y = yi, col=factor(Ecosystem)))+
      geom_point()
#dev.off()

#=============================================================================
# 5.Import geospatial environmental data and join to original data based on lat,lon: 
#
# 1. Soil data from SoilGrids (via soilDB library)
# 2. Average climate characteristics from WorldClim
#=============================================================================
locations_use = data.frame(id = dfa_B_con2$studyGroup ,lat=(dfa_B_con2$lat), lon=(dfa_B_con2$lon),stringsAsFactors = FALSE )
locations = locations_use[!is.na(locations_use[,2]),]

#Soil charactertistics
#####>>>>>>>>>>>>>>>Comment/uncomment here as needed <<<<<<<<<<<<<<<<<<<<
#soil_use = get_SoilGrids(locations)
# save(file = "./data/soilgrids1.var", soil_use)
 load("./data/soilgrids1.var")
#####>>>>>>>>>>>>>>>Comment/uncomment here as needed <<<<<<<<<<<<<<<<<<<<

#Append these to the df
soil_use_key = soil_use[!duplicated(soil_use$studyGroup),]
dfa_B_con2 = left_join(dfa_B_con2, soil_use_key, by = "studyGroup" )

#Average climate variables
#####>>>>>>>>>>>>>>>Comment/uncomment here as needed <<<<<<<<<<<<<<<<<<<<
#climate_use = get_climate(locations)
# colnames(climate_use)[21] = "studyGroup"
# save(file = "./data/worldclim_bio.var", climate_use)
load("./data/worldclim_bio.var")
#####>>>>>>>>>>>>>>>Comment/uncomment here as needed <<<<<<<<<<<<<<<<<<<<

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
#  6.Fit RandomForests models with the environmental covariates. Use a boostrap
#   approach to generate distriubtions of variable importance.Now that we have these environmental covariates, explore some RandomForests 
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

#Why is ocdmean missing entries? 
dfa_ml = dfa_ml [,-11]
#####Fit two versions of the model: One without, then one with the 
#####the categoricals.
# 1. Continuous values only:
#Get a distriubtion of RF model fits
rf_dist = get_RF_dist(dfa_ml)
#Get the average and sd of each variable's ranked position to see how stable
#the importance of each variable is. 
import_ranks = get_rank_importance(rf_dist[[1]])

#####
#2. With the categorical variables
#Combine both data sets
dfa_ml = cbind(dfa_ml,dfa_cat_new)

#Get a distriubtion of RF model fits
rf_dist_cat = get_RF_dist(dfa_ml)
#Get the average and sd of each variable's ranked position to see how stable
#the importance of each variable is. 
import_ranks_cat = get_rank_importance(rf_dist_cat[[1]])

#########
#Plotting
# fig.name = paste("./data/variable_importance2",".pdf",sep="")
# pdf(file=fig.name, height=8, width=8, onefile=TRUE, family='Helvetica', pointsize=16)

p1 = ggplot(import_ranks, aes(x = factor(IncNodePurity), y = IncNodePurity)) +
  geom_point() + 
  geom_errorbar(aes(ymin = IncNodePurity - sd, ymax = IncNodePurity + sd), width = 0.2) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  scale_x_discrete(labels = import_ranks$var[order(import_ranks$IncNodePurity)] )+
  labs(x = 'Variable', y = 'Mean Rank', title = 'Mean Variable Importance ') #+
  #theme_minimal()

irc_use = import_ranks_cat[1:40,]
p2 = ggplot(irc_use , aes(x = factor(IncNodePurity), y = IncNodePurity)) +
  geom_point() + 
  geom_errorbar(aes(ymin = IncNodePurity - sd, ymax = IncNodePurity + sd), width = 0.2) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  scale_x_discrete(labels = irc_use$var[order(irc_use$IncNodePurity)])+
  labs(x = 'Variable', y = 'Mean Rank') #+
  #theme_minimal()

grid.arrange(p1,p2)

#dev.off()


########This is the full list of the bioclimatic variables from WorldClim: 
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
# 7.Based on the top (e.g. 14) covariates from RF variable importance, do another
#   round of dredge to explore the top environmental covariaties.  
#=============================================================================
#Remove NAs
dfa_ml_use = dfa_B_con2[!is.na(dfa_B_con2$bdodmean), ]
dfa_ME = dfa_ml_use 

colnames(dfa_ME)[95:113] = c("AMT", "MDR", "Iso","SeaT","MaxT","MinT","ART",
  "MTempW", "MTempD","MTempH","MTempC","APr","PrW","PrD","SeaPr","PrWQ",
  "PrDQ", "PrH","PrC") 


#Model with the top 14 covariates from the RandomForest variable importance
model_full = rma.mv(yi,vi, mods = ~bdodmean+wv0033mean+AMT+PrC+SeaPr+cecmean+
                                    PrW+Iso+PrWQ+MDR+claymean+siltmean+
                                    nitrogenmean+APr,
                                    random =   ~ 1 | Study/studyGroup, 
                                    data= dfa_ME, method = "ML")

# model_full = rma.mv(yi,vi, mods = ~bdodmean+MinT+MTempC+PrDQ+PrD+wv1500mean+
#                                     PrW+Iso+PrWQ+MDR+claymean+AMT+
#                                     nitrogenmean+phh2omean,
#                                     random =   ~ 1 | Study/studyGroup, 
#                                     data= dfa_ME, method = "ML")

options(na.action = "na.fail") #Needed to run dredge
eval(metafor:::.MuMIn) #Need this to tell MuMIn how to dredge correctly with metafor

#####>>>>>>>>>>>>>>>Comment/uncomment here as needed <<<<<<<<<<<<<<<<<<<<
#model_dredge = dredge(model_full, trace=2)
#save(file="./data/dredged_RE_model2.var", model_dredge)
load(file="./data/dredged_RE_model2.var")
#####>>>>>>>>>>>>>>>Comment/uncomment here as needed <<<<<<<<<<<<<<<<<<<<

options(na.action = "na.omit") # set back to default

#Examine models With AICc value no more than 2 units away from the best model
best_mods = subset(model_dredge, delta <= 2, recalc.weights=FALSE)

#Multi-model inference
avg_model = summary(model.avg(best_mods))

#Top model
top_model = get.models(model_dredge,subset = 1)[[1]]

#####To export summaries into csv: 
# fp = "./data/best_re_mod.csv"                                                                                                
# sink(file = fp)                                                                                                              
# summary(top_model)                                                                                                           
# sink()    