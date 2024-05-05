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

#=============================================================================
# load data sets and do some initial sanity checks. 
#=============================================================================
#Averill 2022 meta analysis data. Two separate files, representing two 
#different collection stages and potentially different methods? 
dfa1 = read.csv(file="./data/averil_meta1.csv")
dfa2 = read.csv(file="./data/averil_meta2.csv")
#Reads as character because of "?"
dfa1$IsLatestTimePoint = as.numeric(dfa1$IsLatestTimePoint )
#Join all the rows
dfa = full_join(dfa1,dfa2)

#Check dfa size. dim[2] should be bigger than dim(dfa1)[2] and dim(dfa2)[2]
#and also these should all be equal: 
dim(dfa)[2]
dim(dfa2)[2] + length (colnames(dfa1[!(colnames(dfa1) %in% colnames(dfa2))] ))
dim(dfa1)[2] + length (colnames(dfa2[!(colnames(dfa2) %in% colnames(dfa1))] ))



