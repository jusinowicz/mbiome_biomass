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
source("./functions/useful.R")
#=============================================================================
# load data sets and do some initial sanity checks. 
#=============================================================================
#Averill 2022 meta analysis data. Two separate files, representing two 
#different collection stages and potentially different methods? 
dfa1 = read.csv(file="./data/averil_meta1.csv")
dfa2 = read.csv(file="./data/averil_meta2.csv")
use_studies = read.csv(file = "./data/averil_sites.csv")

#Reads as character because of "?"
dfa1$IsLatestTimePoint = as.numeric(dfa1$IsLatestTimePoint )
#Join all the rows
dfa = full_join(dfa1,dfa2)
dfa_av = dfa[dfa$DOI %in% unique(use_studies$DOI), ]

dfa_ind = grep(paste(use_studies$DOI, collapse = "|"), dfa$DOI)


#Check dfa size. dim[2] should be bigger than dim(dfa1)[2] and dim(dfa2)[2]
#and also these should all be equal: 
dim(dfa)[2]
dim(dfa2)[2] + length (colnames(dfa1[!(colnames(dfa1) %in% colnames(dfa2))] ))
dim(dfa1)[2] + length (colnames(dfa2[!(colnames(dfa2) %in% colnames(dfa1))] ))

# Find the coordinates for each site, pull them out, convert all to 
# degree decimal format, add them as new numerical columns. 
coords_temp = t(apply(as.matrix(dfa$location), 1, extract_coordinates))
dfa$lat = coords_temp[,1]
dfa$lon = coords_temp[,2]

#Break data into sets based on the response. Possible responses are: 
# 1. TotBiomass: N = 360, studies = 17
dfa_TB = subset(dfa, !is.na(dfa$TotBiomass) )
dim(dfa_TB)[1]
length(unique(dfa_TB$DOI)) 

# 2. AboveBio: N = 346, studies = 26
dfa_AB = subset(dfa, !is.na(dfa$AboveBio) )
dim(dfa_AB)[1]
length(unique(dfa_AB$DOI)) 

# 3. BelowBio: N = 135, studies = 15
dfa_BB = subset(dfa, !is.na(dfa$BelowBio) )
dim(dfa_BB)[1]
length(unique(dfa_BB$DOI)) 

# 4. Yield N = 0
dfa_Y = subset(dfa, !is.na(dfa$Yield) )
dim(dfa_Y)[1]
length(unique(dfa_Y$DOI)) 

# 5. height: N = 170, studies = 17
dfa_h = subset(dfa, !is.na(dfa$height) )
dim(dfa_h)[1]
length(unique(dfa_h$DOI)) 

# 6. CoverPerc: N = 0 
dfa_CP = subset(dfa, !is.na(dfa$CP) )
dim(dfa_CP)[1]
length(unique(dfa_CP$DOI)) 

# 7. seedlingSurvPerc N = 128, studies = 16
dfa_sSP = subset(dfa, !is.na(dfa$seedlingSurvPerc) )
dim(dfa_sSP)[1]
length(unique(dfa_sSP$DOI)) 

# 8. LeafArea.cm.2.	N = 10, studies = 1
dfa_LA = subset(dfa, !is.na(dfa$LeafArea.cm.2.) )
dim(dfa_LA)[1]
length(unique(dfa_LA$DOI)) 

# 9. LeafPhosphorous.mg.kgDryWeight. N = 417, studies = 44
dfa_LP = subset(dfa, !is.na(dfa$LeafPhosphorous.mg.kgDryWeight.) )
dim(dfa_LP)[1]
length(unique(dfa_LP$DOI)) 

#10. BasalAreaGrowthSeedlings.mm.2. N = 15, studies = 1 
dfa_BAGS = subset(dfa, !is.na(dfa$BasalAreaGrowthSeedlings.mm.2.) )
dim(dfa_BAGS)[1]
length(unique(dfa_BAGS$DOI))

#11. Stem.diameter N = 12, studies = 1
dfa_Sd = subset(dfa, !is.na(dfa$Stem.diameter) )
dim(dfa_Sd)[1]
length(unique(dfa_Sd$DOI))

#12. Germination. N = 3, studies = 1
dfa_G = subset(dfa, !is.na(dfa$Germination.) )
dim(dfa_G)[1]
length(unique(dfa_G$DOI))

#13. NumberofGerminatedSeedlings N = 5, stuies = 1
dfa_ngs= subset(dfa, !is.na(dfa$NumberofGerminatedSeedlings) )
dim(dfa_ngs)[1]
length(unique(dfa_ngs$DOI))
