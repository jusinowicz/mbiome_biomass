#==============================================================================
# For the meta analysis and database: 
# This is STEP 4 in the pipeline:
# 
# Use NLP and a custom NER model to extract the TREATMENTs and RESPONSEs from 
# the text of the Methods and Results sections in scientfic papers. 
#
# This is meant to be the 1st step in paper parsing, trying to glean info from 
# the text before trying more complex table-extraction and figure-extraction
# methods. 
# 
# Current NER: custom_web_ner_abs_v382
#
# The final output is a df/CSV with columns STUDY, TREATMENT, RESPONSE, CARDINAL,
# PERCENTAGE, SENTENCE, ISTABLE. 
# There are separate columns for CARDINAL (a numeric)
# response) and PERCENTAGE because the NER recognizes them separately. This is 
# useful because it helps determine whether actual units of biomass response are 
# being identified or the ratio of response (percentage). 
#
# SENTENCE is the sentence that was parsed for the information in the table 
# ISTABLE is a numeric ranking from 0-2 which suggests how likely it is that the
# the parsed information came from a table that the pdf-parsing grabbed. In this
# case, the results are most definitely not to be trusted. 
#
# This table is meant to help determine what information is available in the paper
# and indicate whether further downstream extraction is necessary. 
#
# See extrat_abstract_dat for installation notes.  
#
# This code works fairly well now, but further downstream processing coul be 
# implemented to help human eyes interpret and sift through the useful information.
# In partiular, removing (or at least flagging) entries that appear to be numbers 
# grabbed from summary statistics (e.g. p-values, F-values, AIC, etc.). This seems 
# to happen frequently. 
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

#|Acknowledgments|References|Bibliography
def preprocess_text(text):
    # Remove References/Bibliography and Acknowledgements sections
	text = re.sub(r'\bREFERENCES\b.*', '', text, flags=re.DOTALL | re.IGNORECASE)
	text = re.sub(r'\bACKNOWLEDGEMENTS\b.*', '', text, flags=re.DOTALL | re.IGNORECASE)
	text = re.sub(r'\bBIBLIOGRAPHY\b.*', '', text, flags=re.DOTALL | re.IGNORECASE)

	# Remove hyphenation at line breaks and join split words
	text = re.sub(r'-\n', '', text)

	# Normalize whitespace (remove multiple spaces and trim)
	#text = re.sub(r'\s+', ' ', text).strip()

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
    sections = {'abstract','introduction','methods','results','discussion','acknowledgments' }
    # Initialize the sections dictionary with each section name as a key and an empty list as the value
    sections = {section: [] for section in sections}
    current_section = None
    
    # Enhanced regex to match section headers
    section_header_pattern = re.compile(r'\b(Abstract|Introduction|Methods|Materials and Methods|Results|Discussion|Conclusion|Background|Summary)\b', re.IGNORECASE)
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

#Function just to find ancestors of a token. 
def get_ancestors(token):
    ancestors = []
    while token.head != token:
        ancestors.append(token.head)
        token = token.head
    return ancestors

# Function to find shortest path between two tokens in the
# dependency tree based on the distance to a common ancestor
# (least common ancestor, LCA)
def find_shortest_path(token1, token2):
    ancestors1 = get_ancestors(token1)
    ancestors2 = get_ancestors(token2)
    ancestors2.insert(0,token2)
    #print(f"Ancestors 1 {ancestors1}")
    #print(f"Ancestors 2 {ancestors2}")
    # Find the lowest common ancestor
    common_ancestor = None
    for ancestor in ancestors1:
        if ancestor in ancestors2:
            common_ancestor = ancestor
            break
    if common_ancestor is None:
        return float('inf')
    # Calculate the distance as the number of nodes in the dependency tree
    #print(f"Common ancestor {common_ancestor}")
    distance1 = ancestors1.index(common_ancestor) + 1
    distance2 = ancestors2.index(common_ancestor) + 1
    #print(f"Distance1 = {distance1} and Distance2 = {distance2}")
    distance = distance1 + distance2
    return distance

#Function to trace syntactical dependency back to a specific label
#Use this to find the TREATMENT corresponding to a CARDINAL or PERCENTAGE
#If you want to see the tree for a specific token use the print_tree
#function defined below.
def find_label_in_tree(token, label_id):
	vnames = []
	level = 0
	for ancestor in token.ancestors:
		print(f"ancestor: {ancestor}, all ancestors {list(token.ancestors)}")
		for child in ancestor.children:
			print(f"child: {child}, , all children {list(ancestor.children)}")
			if child.ent_type_ in label_id:
				vname = child.text.strip(',')
				vnames.append(vname)
				print(f"Names so far: {vnames}")
			elif child.dep_ in ['nmod','nummod','conj', 'appos']:
				print(f"Else if, next tree: {ancestor}")
				find_label_in_tree(ancestor, label_id)
		level += 1
		print(f"level {level}")
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
			entities2 = [ent for ent in resp_span.sent.ents if ent.label_ in label_id]
			for token in resp_span.root.head.subtree:
				#Check it's a type we want, and not punctuation
				if token.ent_type_ in ['CARDINAL', 'PERCENTAGE'] and token.text not in ['%', ' ', ',']:
					value = token.text
					ent1 = next((ent for ent in resp_span.sent.ents if token in ent), None)
					#Find the connected treatment by parsing dependencies
					shortest_distance = float('inf')
					treat = None
					for ent2 in entities2:
						distance = find_shortest_path(ent1.root, ent2.root)
						distance2 = abs(ent2.root.i)
						#Handle the case of equal distances separately
						if distance < shortest_distance:
							shortest_distance = distance
							shortest_distance2 = distance2
							treat = ent2
							#print(f"{treat}, {shortest_distance}")
						#If dependence distances are equal, use whichever precedes the number
						elif distance == shortest_distance:
							if distance2 < shortest_distance2:
								shortest_distance = distance
								shortest_distance2 = distance2
								treat = ent2
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
	# Remove remaining newline characters
	results_text = re.sub(r'\n', ' ', results_text)
	results_doc, results_entities = extract_entities(results_text)
	table = create_table(results_doc, results_entities, study_id)
	data.append(table)


flattened_data = [item for sublist in data for item in sublist]
df = pd.DataFrame(flattened_data)

# Export DataFrame to a CSV file
df.to_csv('./output/extract_from_text2.csv', index=False)

# Export DataFrame to a CSV file
new_df = df[df["ISTABLE"] == 0] 
new_df.to_csv('extract_correct_text1.csv', index=False)


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
from spacy import displacy
html = displacy.render(sentence, style="dep", page=True)
with open("./output/syntactic_tree_ex4.html", "w", encoding="utf-8") as file:
    file.write(html)

# Generate the dependency tree in html
# displacy.render(doc, style="dep", options={"compact": True, "color": "blue"})
# tree = displacy.render(sentence, style="dep", options={"compact": True, "color": "blue"})


s1 = "The plant dry weight was improved with the application of Bradyrhizobium by 59.3, 13.5, and 34.8%; and with the application of AMF by 63.2, 21.8, and 41.0% and with their combination by 61.7, 18.7, 38.7% in both growing seasons as compare with control, 100% NPK and 50% NPK respectively(Table 2)."
s2 = "The applications of fertilizer and AMF increased the dry weight by 100 and 300%, respecticely."
s3 = "The application of fertilizer increased the dry weight by 100%, while the application of AMF increased the dry weight by 300%."
s4 = "The highest dry biomass shoot found was 10.39 g  and root 9.59 g/plant in T3 inoculated with AMF in  T. arjuna, which was 29.71% and 19.72% higher  compared to non-inoculated control plants grown in  the same ratio of soil (Table 2)."



