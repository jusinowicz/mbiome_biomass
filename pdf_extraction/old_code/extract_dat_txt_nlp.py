#==============================================================================
# Attempt to automate extraction of treatment effects from scientific papers, 
# along with covariates or other descriptors of interes. 
#
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

#PDF extraction
import fitz  # PyMuPDF
#Text preprocessing
import re
import nltk
from nltk.tokenize import sent_tokenize
#NER and NLP
import spacy

#Step 1: Extract Text from PDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

pdf_path = './papers/Semchenko2019.pdf'
pdf_text = extract_text_from_pdf(pdf_path)

#Step 2: Preprocess Text
# Download NLTK data files
nltk.download('punkt')

def preprocess_text(text):
    # Remove References section
    text = re.sub(r'\bREFERENCES\b.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    return sentences

sentences = preprocess_text(pdf_text)

#Step 3: Identify Sections

#Define a mapping for section variations
section_mapping = {
    'abstract': 'abstract',
    'introduction': 'introduction',
    'background': 'introduction',
    'methods': 'methods',
    'materials and methods': 'methods',
    'methodology': 'methods',
    'experimental methods': 'methods',
    'results': 'results',
    'findings': 'results',
    'discussion': 'discussion',
    'conclusion': 'conclusion'
}

def identify_sections(sentences):
	sections = {}
	current_section = None
    
	# Enhanced regex to match section headers
	section_header_pattern = re.compile(r'\b(Abstract|Introduction|Methods|Materials and Methods|Results|Discussion|Conclusion|Background|Summary|Acknowledgments|References|Bibliography)\b', re.IGNORECASE)
	for sentence in sentences:
		# Check if the sentence is a section header
		header_match = section_header_pattern.search(sentence)
		if header_match:
			section_name = header_match.group(1).lower()
			normalized_section = section_mapping.get(section_name, section_name)
			current_section = normalized_section
			sections[current_section] = []
			print(f"Matched Section Header: {header_match}")  # Debugging line
		elif current_section:
			sections[current_section].append(sentence)
	return sections

sections = identify_sections(sentences)


#Step 4: Named Entity Recognition (NER)

# Load pre-trained model from spaCy
#nlp = spacy.load("en_core_sci_md")

#Load custom NER 
nlp = spacy.load("custom_web_ner_abs_v381")

def extract_entities(text):
	doc = nlp(text)
	#This line is for extracting entities only
	#entities = [(ent.text, ent.label_) for ent in doc.ents]
	#This line is for extracting entities with dependencies. 
	entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]
	return doc,entities

# Apply NER to the Methods section to identify treatments and covariates
abstract_text = " ".join(sections.get('abstract', []))
doc, entities = extract_entities(abstract_text)
print(entities)

methods_text = " ".join(sections.get('methods', []))
doc, entities = extract_entities(methods_text)
print(entities)

#Group and refine the treatment groups
def find_treatment_groups(doc, entities):
	# Create a dictionary mapping token indices to entities
	entity_dict = {ent[2]: (ent[0], ent[1]) for ent in entities}
	treatment_groups = []
	for sent in doc.sents:
		sent_treatments = []
		sent_entities = {token.i: entity_dict[token.i] for token in sent if token.i in entity_dict}
		for token in sent:
			if token.i in sent_entities and sent_entities[token.i][1] == 'TREATMENT':
				treatment = [sent_entities[token.i][0]]
				# Find modifiers or compound parts of the treatment entity using dependency parsing
				for child in token.children:
					if child.dep_ in ['amod', 'compound', 'appos', 'conj', 'advmod', 'acl', 'prep', 'pobj', 'det']:
						if child.i in sent_entities:
							treatment.append(sent_entities[child.i][0])
						else:
							treatment.append(child.text)
				# Also check if the token itself has a head that is a treatment (e.g., 'home' as in 'home soil biota')
				if token.head.i in sent_entities and sent_entities[token.head.i][1] == 'TREATMENT' and token.head != token:
					treatment.append(token.head.text)
				# Sort and join treatment parts to maintain a consistent order
				treatment = sorted(treatment, key=lambda x: doc.text.find(x))
				sent_treatments.append(" ".join(treatment))
		if sent_treatments:
			treatment_groups.extend(sent_treatments)
	# Removing duplicates and returning the result
	return list(set(treatment_groups))

# Find treatment groups
treat_matches = find_treatment_groups(doc, entities)
print(treat_matches)

#Group and refine the treatment groups
def find_inoc_groups(doc, entities):
	# Create a dictionary mapping token indices to entities
	entity_dict = {ent[2]: (ent[0], ent[1]) for ent in entities}
	inoc_groups = []
	for sent in doc.sents:
		sent_inocs = []
		sent_entities = {token.i: entity_dict[token.i] for token in sent if token.i in entity_dict}
		for token in sent:
			if token.i in sent_entities and sent_entities[token.i][1] == 'INOCTYPE':
				inoc = [sent_entities[token.i][0]]
				# Find modifiers or compound parts of the inoc entity using dependency parsing
				for child in token.children:
					if child.dep_ in ['amod', 'compound', 'appos', 'conj', 'advmod', 'acl', 'prep', 'pobj', 'det']:
						if child.i in sent_entities:
							inoc.append(sent_entities[child.i][0])
						else:
							inoc.append(child.text)
				# Also check if the token itself has a head that is a inoc (e.g., 'home' as in 'home soil biota')
				if token.head.i in sent_entities and sent_entities[token.head.i][1] == 'INOCTYPE' and token.head != token:
					inoc.append(token.head.text)
				# Sort and join inoc parts to maintain a consistent order
				inoc = sorted(inoc, key=lambda x: doc.text.find(x))
				sent_inocs.append(" ".join(inoc))
		if sent_inocs:
			inoc_groups.extend(sent_inocs)
	# Removing duplicates and returning the result
	return list(set(inoc_groups))

# Find inoc groups
inoc_matches = find_inoc_groups(doc, entities)
print(inoc_matches)

# def find_treatment_groups(doc, entities):
# 	# Create a dictionary mapping token indices to entities
# 	entity_dict = {ent[2]: (ent[0], ent[1]) for ent in entities}
# 	treatment_groups = []
# 	for sent in doc.sents:
# 		sent_treatments = []
# 		sent_entities = {token.i: entity_dict[token.i] for token in sent if token.i in entity_dict}
# 		for token in sent:
# 			if token.i in sent_entities and sent_entities[token.i][1] == 'TREATMENT':
# 				treatment = [sent_entities[token.i][0]]
# 				# Find modifiers or compound parts of the treatment entity using dependency parsing
# 				for child in token.children:
# 					if child.dep_ in ['amod', 'compound', 'appos', 'conj', 'advmod', 'acl', 'prep', 'pobj', 'det']:
# 						if child.i in sent_entities:
# 							treatment.append(sent_entities[child.i][0])
# 						else:
# 							treatment.append(child.text)
# 				# Sort and join treatment parts to maintain a consistent order
# 				treatment = sorted(treatment, key=lambda x: token.i)
# 				sent_treatments.append(" ".join(treatment))
# 		if sent_treatments:
# 			treatment_groups.extend(sent_treatments)
# 	return treatment_groups


# def find_treatment_groups(doc, entities):
# 	# Dictionary mapping entity start indices to entities
# 	entity_dict = {ent[2]: (ent[0], ent[1]) for ent in entities}  
# 	treatment_groups = []
# 	for sent in doc.sents:
# 		sent_entities = [entity_dict[token.i] for token in sent if token.i in entity_dict]
# 		sent_treatments = []
# 		for token in sent:
# 			if token.i in entity_dict and entity_dict[token.i][1] == 'TREATMENT':
# 				treatment = [entity_dict[token.i][0]]
# 				# Find tokens related to the treatment using dependency parsing
# 				for child in token.children:
# 					if child.i in entity_dict and entity_dict[token.i][1] in ['INOCTYPE', 'TREATMENT']:
# 						treatment.append(entity_dict[child.i][0])
# 				sent_treatments.append(" ".join(treatment))
# 		if sent_treatments:
# 			treatment_groups.extend(sent_treatments)
# 	return treatment_groups

# def find_treatment_groups(doc, entities):
# 	entity_dict = {ent[2]: (ent[0], ent[1], ent) for ent in entities}  # Dictionary mapping start indices to entities
# 	treatment_groups = []
# 	for sent in doc.sents:
# 		sent_entities = [entity_dict[token.i] for token in sent if token.i in entity_dict]
# 		sent_treatments = []
# 		for entity_text, entity_label, entity_obj in sent_entities:
# 			if entity_label == 'TREATMENT':
# 				treatment = [entity_text]
#                 # Use spaCy's dependency parsing to find related entities
# 				if isinstance(entity_obj, spacy.tokens.Token):  # Check if entity_obj is a single token
# 					subtree = [entity_obj]  # Handle single token case
# 				else:
# 					# Check if entity_obj has a subtree attribute (for spans)
# 					if hasattr(entity_obj, 'subtree'):
# 						subtree = list(entity_obj.subtree)  # Get entire subtree
# 					else:
# 						subtree = [entity_obj]  # Default to entity_obj if no subtree
# 				for token in subtree:
# 				# for token in subtree:
# 				# 	if token[2] in entity_dict and entity_dict[token[2]][1] in ['INOCTYPE', 'TREATMENT']:
# 				# 		treatment.append(entity_dict[token[2]][0])
# 					if token.i in entity_dict and entity_dict[token.i][1] in ['INOCTYPE', 'TREATMENT']:
# 						treatment.append(entity_dict[token.i][0])
# 				sent_treatments.append(" ".join(treatment))
# 		if sent_treatments:
# 			treatment_groups.extend(sent_treatments)
# 	return treatment_groups

# def find_treatment_groups(doc, entities):
# 	entity_dict = {ent[2]: (ent[0], ent[1], ent) for ent in entities}  # Dictionary mapping start indices to entities
# 	treatment_groups = []
# 	for sent in doc.sents:
# 		sent_entities = [entity_dict[token.i] for token in sent if token.i in entity_dict]
# 		sent_treatments = []
# 		for entity_text, entity_label, entity_obj in sent_entities:
# 			if entity_label == 'TREATMENT':
# 				treatment = [entity_text]
# 				for child in entity_obj.children:
# 					if child.i in entity_dict and (entity_dict[child.i][1] in ['INOCTYPE', 'TREATMENT']):
# 						treatment.append(entity_dict[child.i][0])
# 				sent_treatments.append(" ".join(treatment))
# 		if sent_treatments:
# 			treatment_groups.extend(sent_treatments)
# 	return treatment_groups

# def find_treatment_groups(doc, entities):
# 	entity_dict = {ent[2]: (ent[0], ent[1], ent) for ent in entities}  # Dictionary mapping start indices to entities
# 	treatment_groups = []
# 	for token in doc:
# 		if token.i in entity_dict:
# 			entity_text, entity_label, entity_obj = entity_dict[token.i]
# 			if entity_label == 'TREATMENT':
# 				treatment = [entity_text]
# 				for child in token.children:
# 					if child.i in entity_dict:
# 						child_text, child_label, _ = entity_dict[child.i]
# 						if child_label in ['INOCTYPE', 'TREATMENT']:
# 							treatment.append(child_text)
# 				treatment_groups.append(" ".join(treatment))
# 	return treatment_groups

# # Find treatment groups
# treatment_groups = find_treatment_groups(doc, entities)
# print(treatment_groups)

# Extract numerical entities with dependency parsing
def find_dependencies(doc, entities):
	entity_dict = {ent[2]: (ent[0], ent[1]) for ent in entities}  # Dictionary mapping start indices to entities
	matches = []
	for token in doc:
		if token.i in entity_dict:
			entity_text, entity_label = entity_dict[token.i]
			if entity_label in ['CARDINAL', 'PERCENTAGE']:
				# Check children for related entities
				for child in token.children:
					if child.i in entity_dict:
						child_text, child_label = entity_dict[child.i]
						if child_label in ['TREATMENT', 'RESPONSE']:
							matches.append((entity_text, child_text, child_label))
						elif child_label in ['INOCTYPE']:
							matches.append((entity_text, child_text, child_label))
			elif entity_label in ['TREATMENT', 'RESPONSE']:
				# Check children for related entities
				for child in token.children:
					if child.i in entity_dict:
						child_text, child_label = entity_dict[child.i]
						if child_label in ['CARDINAL', 'PERCENTAGE']:
							matches.append((child_text, entity_text, entity_label))
							# Check parent for related entities
			if token.head.i in entity_dict:
				head_text, head_label = entity_dict[token.head.i]
				if (entity_label in ['CARDINAL', 'PERCENTAGE'] and head_label in ['TREATMENT', 'RESPONSE']) or \
					(entity_label in ['TREATMENT', 'RESPONSE'] and head_label in ['CARDINAL', 'PERCENTAGE']):
					matches.append((entity_text, head_text, head_label))
	return matches

# Find dependencies and matches
num_matches = find_dependencies(doc, entities)
print(matches)

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# Open PDF and iterate through pages
pdf_document = fitz.open(pdf_path )

for page_number in range(len(pdf_document)):
	page = pdf_document.load_page(page_number)
	pix = page.get_pixmap()
	img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
	text = pytesseract.image_to_string(img)
	print(f"Text from page {page_number}:\n", text)

import tabula
pdf_path = './papers/35410135.pdf' 
# Extract tables from a PDF file
tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)

# Process each extracted table
for i, df in enumerate(tables):
    print(f"Table {i}:\n", df)


import camelot
# Extract tables from PDF
tables = camelot.read_pdf(pdf_path, pages='all', flavor = 'stream', column_tol=2)
table = tables[9].df.values.tolist() #Converts to text format
#This gets rid of the first row, which seems to help because camelot will return a lot of noise
table_df = pd.DataFrame(table[1:], columns=table[0]) 
#Make sure each cell in the new df is actually of type str
table_df = table_df.astype(str)  

# Load the tokenizer and model
tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

# Tokenize inputs
inputs = tokenizer(table=table_df, queries=queries, padding="max_length", return_tensors="pt")

# Get model predictions
outputs = model(**inputs)
logits = outputs.logits

# Detach logits from computation graph
logits = logits.detach()

# Decode answer
predicted_answer_coordinates, _ = tokenizer.convert_logits_to_predictions(inputs, logits)

# Debugging: print the predicted coordinates
print("Predicted answer coordinates:", predicted_answer_coordinates)

# Initialize the predicted_answer list
predicted_answer = []

# Convert the coordinates to text manually
for coordinates_list in predicted_answer_coordinates[0]:  # Extract the list of coordinates
    for coordinates in coordinates_list:  # Extract each tuple of coordinates
        try:
            row, column = coordinates  # Unpack the row and column indices
            answer = table_df.iat[row, column]
            predicted_answer.append(answer)
        except IndexError:
            predicted_answer.append("IndexError")  # Handle any unexpected indexing issues

print(predicted_answer)


dfs=[]
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            df = pd.DataFrame(table[1:], columns=table[0])
            print(df)

class PDFTables:
	def __init__(self):
		self.tables = []
	def add_table(self, table):
		self.tables.append(table)
	def get_tables(self):
		return self.tables
	def to_dict(self):
		return {f"Table {i+1}": table.to_dict() for i, table in enumerate(self.tables)}
	def to_excel(self, filename):
		with pd.ExcelWriter(filename) as writer:
			for i, table in enumerate(self.tables):
				table.to_excel(writer, sheet_name=f"Table {i+1}")


pdf_tables = PDFTables()
with pdfplumber.open(pdf_path) as pdf:
	for page_number, page in enumerate(pdf.pages):
		print(f"Processing page {page_number + 1}...")
		tables = page.extract_table()
		print(f"Found {len(tables)} tables on page {page_number + 1}")
		for table_index, table in enumerate(tables):
			if table:  # Check if table is not empty
				print(f"Processing table {table_index + 1} on page {page_number + 1}")
				df = pd.DataFrame(table[1:], columns=table[0])
				pdf_tables.add_table(df)
			else:
				print(f"Table {table_index + 1} on page {page_number + 1} is empty or could not be processed")
    
    return pdf_tables

table_settings = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_y_tolerance": 5,
    "intersection_x_tolerance": 15,
}

from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import torch


# Example table as a pandas DataFrame
data = {
    "Inoculant": ["Inoculated", "Uninoculated"],
    "Biomass": [3.5, 2.1]
}
table = pd.DataFrame(data)

# Ensure all data in the DataFrame are strings
table = table.astype(str)

# Example questions
queries = ["What is the biomass of inoculated soil?"]

# Tokenize inputs
inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")

# Get model predictions
outputs = model(**inputs)
logits = outputs.logits

# Detach logits from computation graph
logits = logits.detach()

# Decode answer
predicted_answer_coordinates, _ = tokenizer.convert_logits_to_predictions(inputs, logits)

# Debugging: print the predicted coordinates
print("Predicted answer coordinates:", predicted_answer_coordinates)

# Initialize the predicted_answer list
predicted_answer = []

# Convert the coordinates to text manually
for coordinates_list in predicted_answer_coordinates[0]:  # Extract the list of coordinates
    for coordinates in coordinates_list:  # Extract each tuple of coordinates
        try:
            row, column = coordinates  # Unpack the row and column indices
            answer = table.iat[row, column]
            predicted_answer.append(answer)
        except IndexError:
            predicted_answer.append("IndexError")  # Handle any unexpected indexing issues

print(predicted_answer)  # Output should be the answer(s) based on table 

###################################
# Create a list of unique TREATMENT and INOCTYPE entities
unique_treatments = list(set([ent[0] for ent in entities if ent[1] == 'TREATMENT']))
unique_inoc_types = list(set([ent[0] for ent in entities if ent[1] == 'INOCTYPE']))

# Create a DataFrame with columns for each label category
columns = ['TREATMENT', 'INOCTYPE', 'ECOREGION', 'ECOTYPE', 'LANDUSE', 'LOCATION', 'RESPONSE', 'SOILTYPE']
df = pd.DataFrame(columns=columns)

# Populate the DataFrame
rows = []
for treatment in unique_treatments:
    for inoc_type in unique_inoc_types:
        row = {'TREATMENT': treatment, 'INOCTYPE': inoc_type}
        rows.append(row)

df = pd.DataFrame(rows, columns=columns)

print(df)



def contextual_match(text, entities):
	# Find CARDINAL and PERCENTAGE entities
	cardinal_entities = [(ent.text, ent.start_char, ent.end_char) for ent in nlp(text).ents if ent.label_ == 'CARDINAL']
	percentage_entities = [(ent.text, ent.start_char, ent.end_char) for ent in nlp(text).ents if ent.label_ == 'PERCENTAGE']
	matches = []
	for cardinal in cardinal_entities:
		context = text[max(0, cardinal[1] - 50): min(len(text), cardinal[2] + 50)]
		for ent in entities:
			if ent[1] in ['TREATMENT', 'RESPONSE'] and ent[0] in context:
				matches.append((cardinal[0], ent[0], ent[1]))
	for percentage in percentage_entities:
		context = text[max(0, percentage[1] - 50): min(len(text), percentage[2] + 50)]
		for ent in entities:
			if ent[1] in ['TREATMENT', 'RESPONSE'] and ent[0] in context:
				matches.append((percentage[0], ent[0], ent[1]))
	return matches

# Find contextual matches
matches = contextual_match(abstract_text, entities)
print(matches)




#Step 5: Pattern Matching and Keyword Extraction
def extract_keywords(text, keywords):
	keyword_sentences = []
	for keyword in keywords:
		matches = re.findall(rf'\b{keyword}\b', text, re.I)
		keyword_sentences.extend(matches)
	return keyword_sentences

treatment_keywords = treat_matches ['treatment', 'therapy', 'drug', 'intervention']
response_keywords = ['response', 'outcome', 'result', 'effect']
covariate_keywords = ['covariate', 'variable', 'factor']

treatment_sentences = extract_keywords(abstract_text, treatment_keywords)
response_sentences = extract_keywords(methods_text, response_keywords)
covariate_sentences = extract_keywords(methods_text, covariate_keywords)

print("Treatments:", treatment_sentences)
print("Responses:", response_sentences)
print("Covariates:", covariate_sentences)

#Step 6: Contextual Analysis

def analyze_context(sentences, entities):
	context_info = {}
	for sentence in sentences:
		for entity in entities:
			if entity in sentence:
				if entity not in context_info:
					context_info[entity] = []
				context_info[entity].append(sentence)
	return context_info

treatment_context = analyze_context(sentences, treatment_sentences)
response_context = analyze_context(sentences, response_sentences)
covariate_context = analyze_context(sentences, covariate_sentences)

print("Treatment Context:", treatment_context)
print("Response Context:", response_context)
print("Covariate Context:", covariate_context)

treatments = [ent[0] for ent in entities if ent[1] == "TREATMENT"]
responses = [ent[0] for ent in entities if ent[1] == "RESPONSE"]
treatment_response_pairs = list(zip(treatments, responses))


pattern = re.compile(r'\b(Abstract|Introduction|Methods?|Materials and Methods?|Results?|Discussion|Conclusion|Background|Summary|Acknowledgments|References|Bibliography)\b', re.IGNORECASE)
for i, sentence in enumerate(sentences):
	if pattern.search(sentence):
		header_match = pattern.search(sentence)
		introduction_index = i
		#break  # Stop searching after the first occurrence

for i, sentence in enumerate(sentences):
	header_match = pattern.search(sentence)
	print(f"Matched Section Header: {header_match}")  # Debugging line
	#break  # Stop searching after the first occurrence



for sentence in sentences:
	header_match = section_header_pattern.search(sentence)
	if section_header_pattern.search(sentence):
		current_section = header_match.group(1).strip().lower()
		sections[current_section] = []
		print(f"Matched Section Header: {current_section}")  # Debugging line