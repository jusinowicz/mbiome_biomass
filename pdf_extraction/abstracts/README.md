# Soil microbiomes and biomass meta analysis project
## This is the abstract parsing and NER training arm
1. **meta_analyze_get.py**: This is STEP 1 and STEP1C in the pipeline.

- STEP1: Automate identification and retrieval of citations, abstracts for potentially relevant papers based on keyword search via PubMed. Automate creation (if needed) and uploading of abstracts to the labeling project in Label Studio (currently mbb_abstracts). This is done to help train the custom NER.

- STEP1C: Use the current version of the NER to streamline labeling by generating predictions. The labeling process is iterative! Label, predict, correct, generate a new version of the NER (via meta_analyze_model_update.py), use it to label, predict...

2. **meta_analyze_model_update.py**: This is STEP 1B in the pipeline:
Load the annotated data from Label Studio into python, clean, convert it, and then fit the NER with spacy. Fitting the NER can either be done from scratch, or by loading the custom NER and training it on new labels. 

3. **extract_abstract_dat.py**: # This is STEP 2 in the pipeline:
Use NLP and a custom NER model to extract information from scientific abstracts for following key categories: 

- TREATMENT: Could be any number of inoculation, nutrient, environmental
		 experimental protocols.  
- INOCTYPE: The type of inoculant. Could be species, group (e.g. AMF), or more generic (e.g. soil biota)
- RESPONSE: Should be either biomass or yield 
- SOILTYPE: The type of soil
- FIELDGREENHOUSE: Is the experiment performed in a greenhouse or field
- LANDUSE: For experiments where the context is a history of land use, e.g. agriculture, urban, disturbance (pollution, mining) 
- ECOTYPE: Could be true ecotype (e.g. wetlands, grasslands, etc.) or a single species in the case of ag studies (e.g. wheat)
- ECOREGION: Reserved for very broad categories. 
- LOCATION: If given, a geographic location for experiment
 
- The code will cycle through a list of abstracts, extract the pertanent information, and either create or add the information to a spreadsheet.

- Each entry in the spreadsheet will actually be a list of possible values. For example, TREATMENT could be a list of "fertilizer, combined inoculation, sterilized soil, AMF..." The resulting spreadsheet is meant to be its own kind of database that a researcher could use to check whether each article (by its DOI) is likely to contain the information they need. The next step in the processing pipeline would be code to use something like regular expression matching to identify these studies from the table created here. 

4. **app_flask_model_abstract.py**: This file is to host the NER as an app using flask so that it can be run with Label Studio.