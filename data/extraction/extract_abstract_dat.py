#==============================================================================
# For the meta analysis and database: 
# This is STEP 2 in the pipeline:
# 
# Use NLP and a custom NER model to extract information from 
# scientific abstracts for following key categories: 
# 
# TREATMENT: Could be any number of inoculation, nutrient, environmental
#			 experimental protocols.  
# INOCTYPE: The type of inoculant. Could be species, group (e.g. AMF), or
#			more generic (e.g. soil biota)
# RESPONSE: Should be either biomass or yield 
# SOILTYPE: The type of soil
# FIELDGREENHOUSE: Is the experiment performed in a greenhouse or field
# LANDUSE: For experiments where the context is a history of land use, 
#			e.g. agriculture, urban, disturbance (pollution, mining) 
# ECOTYPE: Could be true ecotype (e.g. wetlands, grasslands, etc.) or a 
#			single species in the case of ag studies (e.g. wheat)
# ECOREGION: Reserved for very broad categories. 
# LOCATION: If given, a geographic location for experiment
# 
# The code will cycle through a list of abstracts, extract the pertanent 
# information, and either create or add the information to a spreadsheet.
# This code requires the following files to exist: 
#  	Make sure "articles" loaded 	from 	meta_analyze_get.py
#	custom_web_ner_abs_v381		from 	meta_analyze_model_updates.py
#
# Each entry in the spreadsheet will actually be a list of possible values.
# For example, TREATMENT could be a list of "fertilizer, combined inoculation,
# sterilized soil, AMF..." The resulting spreadsheet is meant to be its 
# own kind of database that a researcher could use to check whether each 
# article (by its DOI) is likely to contain the information they need. 
# The next step in the processing pipeline would be code to use something 
# like regular expression matching to identify these studies from the table
# created here. 
#
# Installation notes: 
# Install libraries
# One note on installation. This package, which needs to be installed for NLP: 
# py -3.8-m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
## py -3.8 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
#
# was very finicky. I could only get both it and spacy to install and run 
# on python 3.8, not the current 3.12. It seemed possible maybe with 3.11,3.10.3.9
# but very finicky to set up. 
# 
# My recommendation is to install this first and let it install its own 
# version of spacy and dependencies (something with pydantic versions seems
# to be the problem).
# 
# The package en_core_sci_md also requires that C++ is installed on your system, so visual studio build
# tools on Windows.
#
# py -m pip install PyPDF2 pdfplumber tabula-py jpype1 PyMuPDF Pillow nltk
# For spacy:
# py -m spacy download en_core_web_sm
# For NER: 
#
# py -3.8 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz
# py -3.8 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_craft_md-0.5.0.tar.gz

#==============================================================================
py -3.8

import pandas as pd
import dill #To load saved MetaPubObject list of papers/abstracts
#PDF extraction
import fitz  # PyMuPDF
#Text preprocessing
import re
import nltk
from nltk.tokenize import sent_tokenize
#NER and NLP
import spacy

#Step 1: Open the abstracts. See meta_analyze_get.py if you need to 
#do a search and download abstracts and create this file
with open('articles.pkl', 'rb') as file:
    articles_loaded = dill.load(file)

#Step 2: Named Entity Recognition (NER)

# Load pre-trained model from spaCy
#nlp = spacy.load("en_core_sci_md")

#Load custom NER 
nlp = spacy.load("custom_web_ner_abs_v381")

#Specify the labels to extract
label_list = ["TREATMENT", "INOCTYPE", "RESPONSE","SOILTYPE", "FIELDGREENHOUSE", "LANDUSE", "ECOTYPE", "ECOREGION","LOCATION"]

#This function will return the text and the entities for processing
def extract_entities(text):
	doc = nlp(text)
	#This line is for extracting entities only
	#entities = [(ent.text, ent.label_) for ent in doc.ents]
	#This line is for extracting entities with dependencies. 
	entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]
	return doc,entities


#This function is to group and refine within each entity group
def find_entity_groups(doc, entities, label_type):
	# Create a dictionary mapping token indices to entities of the given label type
	entity_dict = {ent[2]: (ent[0], ent[1]) for ent in entities if ent[1] == label_type}
	entity_groups = []
	for sent in doc.sents:
		sent_entities = {token.i: entity_dict[token.i] for token in sent if token.i in entity_dict}
		sent_entity_groups = []
		for token in sent:
			if token.i in sent_entities and sent_entities[token.i][1] == label_type:
				entity_group = [sent_entities[token.i][0]]
				# Find modifiers or compound parts of the entity using dependency parsing
				for child in token.children:
					if child.dep_ in ['amod', 'compound', 'appos', 'conj', 'advmod', 'acl', 'prep', 'pobj', 'det']:
						if child.i in sent_entities:
							entity_group.append(sent_entities[child.i][0])
						else:
							entity_group.append(child.text)
				# Also check if the token itself has a head that is an entity of the same type
				if token.head.i in sent_entities and sent_entities[token.head.i][1] == label_type and token.head != token:
					entity_group.append(token.head.text)
				# Sort and join entity parts to maintain a consistent order
				entity_group = sorted(entity_group, key=lambda x: doc.text.find(x))
				sent_entity_groups.append(" ".join(entity_group))
		if sent_entity_groups:
			entity_groups.extend(sent_entity_groups)
	# Removing duplicates and returning the result
	return list(set(entity_groups))

# Create a DataFrame with columns for each label category, plus the DOI
columns = ["DOI"]+ label_list

# Initialize a list to collect rows
rows = []
#Cycle through the articles, get the information, and add to the table
for article in articles_loaded:
	# Apply NER to the Abstract to identify treatments and covariates
	abstract_text = article.abstract
	doc, entities = extract_entities(abstract_text)
	#print(entities)
	row = {"DOI": article.doi}
	#Cycle through the labels
	for label in label_list:
		#Find the groups
		entity_matches = find_entity_groups(doc, entities, label)
		row[label] = "; ".join(entity_matches)  # Join matches with a separator, e.g., '; '
	# Collect the row
	rows.append(row)

#Create and save the df
df = pd.DataFrame(rows, columns=columns)
#print(df)
df.to_csv('abstract_parsing1.csv', index=False)