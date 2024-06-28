#==============================================================================
# For the meta analysis and database: 
# This is STEP 3 in the pipeline:
# 
# Use NLP and a custom NER model to extract the TREATMENTs and RESPONSEs from 
# the text of the Methods and Results sections in scientfic papers. 
#
# Current NER: custom_web_ner_abs_v381
#
# Each entry in the spreadsheet will be...
#
# See extrat_abstract_dat for installation notes.  
#
#==============================================================================
py -3.8

import os
import pandas as pd

#PDF extraction
import fitz  # PyMuPDF

#Text preprocessing
import re
import nltk
from nltk.tokenize import sent_tokenize

#NER and NLP
import spacy
#Current NER to use: 
output_dir = "custom_web_ner_abs_v382"

#Post-NER processing
from collections import defaultdict

#==============================================================================
# This section loads the PDFs from a folder and gets the right sections: 
# if new papers are added:
# 1. Import the PDFs
# 2. Preprocess text with NLTK
# 3. Extract Methods and Results sections 
# 4. Upload these to label studio
#==============================================================================
#Step 1: Extract Text from PDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

#Step 2: Preprocess Text
# Download NLTK data files
nltk.download('punkt')

def preprocess_text(text):
    # Remove References section
    text = re.sub(r'\bREFERENCES\b.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    return sentences

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
    'conclusion': 'discussion',
    'summary': 'discussion'
}

def identify_sections(sentences):
    sections = {'abstract','introduction','methods','results','discussion' }
    # Initialize the sections dictionary with each section name as a key and an empty list as the value
    sections = {section: [] for section in sections}
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
            sections[current_section].append(sentence)
            print(f"Matched Section Header: {header_match}")  # Debugging line
        elif current_section:
            sections[current_section].append(sentence)
    return sections

#Step 4. Filter sentences containing specific keywords
#The output is much higher quality if we only focus on sentences which have 
#been labeled as containing RESPONSE variables. 
def filter_sentences(sentences, keywords):
    filtered_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    return filtered_sentences

#Step 5. Run NER to extract entities
#Load the model
nlp = spacy.load(output_dir)
def extract_entities(text):
	doc = nlp(text)
	#This line is for extracting entities with dependencies. 
	entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]
	return doc,entities


#Step 6. Use the dependency parsing of the sentences done by spacy NLPs to 
#infer relationships between RESPONSE, TREATMENT, and CARDINAL/PERCENTAGE to
#ultimately build out a table of these relationships. 

#Function to trace syntactical dependency back to a specific label
#Use this to find the TREATMENT corresponding to a CARDINAL or PERCENTAGE
def find_label_in_tree(token, label_id):
	vnames = []
	for ancestor in token.ancestors:
		for child in ancestor.children:
			if child.ent_type_ in label_id:
				vname = child.text.strip(',')
				vnames.append(vname)
			elif child.dep_ in ['nmod','nummod','conj', 'appos']:
				find_label_in_tree(ancestor, label_id)
	return vnames

#Function using heuristics to guess whether a sentence might actually be 
#a table extracted as a single sentence. Since parsing text is messy, this 
#provides a tool to infer the quality of output coming from a document. 
def from_table(sent):
	"""
	Determine if a given sentence is likely from a table based on heuristic checks.

    Args:
        sent: A spaCy Span object representing the sentence.

    Returns:
        bool: True if the sentence is likely from a table, False otherwise.
	"""
	text = sent.text
	howtrue = 0 #Make this a scale from 0 to MAX

	# Heuristic 1: Check for white space characters used in formatting
	spaces = text.count('\u2009')
	if(spaces>2):
		howtrue +=1

	# Heuristic 2: Check for consistent alignment/spacing
	lines = text.split('\n')
	if len(lines) > 1:
		line_lengths = [len(line.strip()) for line in lines]
		if max(line_lengths) - min(line_lengths) < 10:  # Threshold for alignment
			howtrue += 1
			
	# # Heuristic 3: Check for many newlines denoting tabular format
	tabs = text.count('\n')
	if(tabs > 10):
		howtrue += 1

	return(howtrue)

# Function to create a table of treatments and responses using syntactical
# dependencies within the sentence to infer how numbers and treatments are 
# related. 
def create_table(doc, entities, study_id):
	data = []
	responses = ['dry weight', 'biomass']
	label_id = ["TREATMENT", "INOCTYPE"]
	for response in responses:
		response_ents = [ent for ent in entities if ent[1] == 'RESPONSE' and response in ent[0].lower()]
		for resp_ent in response_ents:
			resp_span = doc[resp_ent[2]:resp_ent[3]]
			for token in resp_span.root.head.subtree:
				#Check it's a type we want, and not punctuation
				if token.ent_type_ in ['CARDINAL', 'PERCENTAGE'] and token.text not in ['%', ' ', ',']:
					value = token.text
					#Find the connected treatment by parsing dependencies
					treat = find_label_in_tree(token, label_id)
					if token.ent_type_ == 'CARDINAL':
						data.append({
							'STUDY': study_id,
							'TREATMENT': treat,
							'RESPONSE': response,
							'CARDINAL': value,
							'PERCENTAGE': '',
							'SENTENCE': token.sent,
							'ISTABLE': from_table(token.sent)
						})
					elif token.ent_type_ == 'PERCENTAGE':
						data.append({
							'STUDY':study_id,
							'TREATMENT': treat,
							'RESPONSE': response,
							'CARDINAL': '',
							'PERCENTAGE': value,
							'SENTENCE': token.sent,
							'ISTABLE': from_table(token.sent)
						})
	#df = pd.DataFrame(data)
	return data


#For debug: pdf_path = "./papers/33888066.pdf"
#pdf_path = "./papers/33893547.pdf"
#pdf_path = "./papers/35410135.pdf"

# Get the list of current PDFs in the directory
pdf_dir = "./papers/" #Where the papers live
new_pdfs = {f for f in os.listdir(pdf_dir) if f.endswith('.pdf')}

# Process the PDFs
data = []
for pdf in new_pdfs:
	#1.
	pdf_path = pdf_dir + pdf
	study_id = pdf.rstrip(".pdf")
	pdf_text = extract_text_from_pdf(pdf_path)
	#2. 
	sentences = preprocess_text(pdf_text)
	#3.
	sections = identify_sections(sentences)
	#4.
	# #Get the methods
	# methods_text = " ".join(sections.get('methods', []))
	# methods_doc, methods_entities = extract_entities(methods_text)
	# data_methods=parse_entities(methods_entities) 
	#Filter sentences in the "Results" section
	keywords = ["biomass", "dry weight", "yield"]
	results_text = filter_sentences(sections["results"], keywords) 
	#Extract entities from filtered text
	results_text = " ".join(results_text)
	results_doc, results_entities = extract_entities(results_text)
	table = create_table(results_doc, results_entities, study_id)
	data.append(table)

flattened_data = [item for sublist in data for item in sublist]
df = pd.DataFrame(flattened_data)

# Export DataFrame to a CSV file
df.to_csv('extract_from_text1.csv', index=False)

#==============================================================================
# This section contains miscellaneous tools for parsing and visualizing the 
# sentence dependency structures. This might be temporary
#==============================================================================
# Generate the dependency tree in a textual format
# Access the specific sentence using doc.sents
sentence_index = 6
sentence = list(results_doc.sents)[sentence_index]

# Function to recursively print the tree structure
def print_tree(token, indent=''):
	print(f"{indent}{token.text} <-- {token.dep_} <-- {token.head.text}")
	for child in token.children:
		print_tree(child, indent + '  ')

for token in sentence:
    print_tree(token)

# Save the syntacticdependency visualization to an HTML file
# html = displacy.render(sentence, style="dep", page=True)
# with open("syntactic_tree.html", "w", encoding="utf-8") as file:
#     file.write(html)

# Generate the dependency tree in html
# displacy.render(doc, style="dep", options={"compact": True, "color": "blue"})
# tree = displacy.render(sentence, style="dep", options={"compact": True, "color": "blue"})