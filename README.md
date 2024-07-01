# Soil microbiomes and biomass meta analysis project
The goal of this project is to use existing studies to: 
1. Build a database of experimental studies that investigate whether the addition of soil microbiome innocula (defined broadly) impact plant biomass, and 
2. Analyze the extent to which experimental outcomes contain signals of ecosystem type or environmental covariates.

This repository is structured to reflect each arm of the project. 

**pdf_extraction**: This folder contains the database-building arm of the project. It is built on a series of python scripts that scrape abstracts from [NCBI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6821292/), train an NER utilizing [spaCy](https://spacy.io/usage/spacy-101) and [label-studio](https://labelstud.io/), import PDFs and parse text, tables, and figures using a variety of packages, and try automate the extractions of TREATMENT and RESPONSE data into tables for the analysis.

**anlysis_in_R**: This folder contains the R scripts to analyze the database, assuming it is available as several csv files. It creates output ranging from figures of variable importance (RandomForests), covariate significance and effect size (mixed-effect models), and various saved models. 

Each project folder contains its own README which further details what is happening on that side of the project. 

