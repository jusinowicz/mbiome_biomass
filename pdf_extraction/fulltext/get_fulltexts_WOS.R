#=============================================================================
# Installation of packages for metagear: 
#	install.packages("BiocManager");
#	BiocManager::install("EBImage")
#	devtools::install_github("daniel1noble/metaDigitise")
#=============================================================================
#For the automated download of PDFs.
# Note: Full functionality of downloading probably requires institutional 
# access. However, I have found that it will still download some available 
# papers. 
#=============================================================================
library(metagear)
library(dplyr)

#path to DOI file. This is currently being generated with PubMedFetcher in 
#Python
doi_path =  "all_DOIs.csv"
#Where to save PDFs: 
save_path = "./../papers/"
#=============================================================================
#Load the doi list
doi_list = read.csv(file = doi_path)

# Create a new column with the save path and ".pdf" appended to each PMID
doi_list = doi_list %>%
  mutate(file_path = paste0(save_path, PMID))

#Loop through the list and download each PDF: 
collectionOutcomes = PDFs_collect(doi_list, DOIcolumn = "DOI", FileNamecolumn = "file_path", quiet = TRUE)