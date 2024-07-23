#==============================================================================
# All of the custom functions that are used across modules are collected here
# 
#==============================================================================
#Libraries
#==============================================================================
#for label studio interactions: upload_task
import requests 

#for fetching and saving docs
import os 
from metapub import PubMedFetcher 
from metapub import FindIt 

#PDF extraction: extract_text_from_pdf
import fitz  # PyMuPDF

#Text preprocessing: preprocess_text, identify_sections, 
import re
import nltk
from nltk.tokenize import sent_tokenize
# Download NLTK data files
nltk.download('punkt')

#for NLP/NER work: extract_entities, find_entity_groups
import spacy 
#==============================================================================
# Functions for dealing with label studio 
#==============================================================================
# Function to upload text data to Label Studio
def upload_task(text, project_id):
	import_url = f'{LABEL_STUDIO_URL}/api/projects/{project_id}/import'
	print("Import URL:", import_url)
	response = requests.post(
		import_url,
		headers={'Authorization': f'Token {API_KEY}'},
		json=[{
			'data': {
				'text': text
			}
		}]
	)
	print("Status Code:", response.status_code)
	print("Response Text:", response.text)
	try:
		response_json = response.json()
		print(response_json)
		return response_json
	except requests.exceptions.JSONDecodeError as e:
		print("Failed to decode JSON:", e)
		return None

#==============================================================================
# Functions for fetching and processing docs
#==============================================================================
# Function to fetch and save full text articles
def get_full_text(articles, save_directory):
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)
    # # Create a PubMedFetcher instance
    # fetcher = PubMedFetcher()
    total_attempted = 0
    total_successful = 0
    for article in articles:
        total_attempted += 1
        try:
            # Get the PMID of the article
            pmid = article.pmid
            # Use FindIt to get the URL of the free open access article full text
            url = FindIt(pmid).url
            if url:
                # Get the full text content
                response = requests.get(url)
                response.raise_for_status()  # Raise an error for bad status codes
                # Create a filename for the article based on its PMID
                filename = f"{pmid}.pdf"
                file_path = os.path.join(save_directory, filename)
                # Save the full text to the specified directory
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                print(f"Downloaded full text for PMID {pmid} to {file_path}")
                total_successful += 1
            else:
                print(f"No free full text available for PMID {pmid}")
        except Exception as e:
            print(f"An error occurred for PMID {pmid}: {e}")
    print(f"Total articles attempted: {total_attempted}")
    print(f"Total articles successfully retrieved: {total_successful}")


#Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

#Preprocess Text
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

def identify_sections(sentences, section_mapping):
    sections = {'abstract','introduction','methods','results','discussion' }
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

#The output is much higher quality if we only focus on sentences which have 
#been labeled as containing RESPONSE variables. 
def filter_sentences(sentences, keywords):
    filtered_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    return filtered_sentences

#==============================================================================
# Functions related to training, loading, and applying the NER based on spacy
#==============================================================================
#Clean Annotations Function
#This handy function converts the JSON format to the correct format
#for spacy, deals with misaligned spans, and removes white space and 
#punctuation in the spans. 

def clean_annotations(data):
    cleaned_data = []
    for item in data:
        text = item['data']['text']
        entities = []
        for annotation in item['annotations']:
            for result in annotation['result']:
                value = result['value']
                start, end, label = value['start'], value['end'], value['labels'][0]
                entity_text = text[start:end]
                # Remove leading/trailing whitespace from entity spans
                while entity_text and entity_text[0].isspace():
                    start += 1
                    entity_text = text[start:end]
                while entity_text and entity_text[-1].isspace():
                    end -= 1
                    entity_text = text[start:end]
                # Check for misaligned entries and skip if misaligned
                if entity_text == text[start:end]:
                    entities.append((start, end, label))
        if entities:
            cleaned_data.append((text, {"entities": entities}))
    return cleaned_data


#This function will return the text and the entities for processing
def extract_entities(text, nlp):
	doc = nlp(text)
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

#==============================================================================
# Use the output of a spacy NER to trace syntactical dependencies and 
# build tables.  
#==============================================================================
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


#==============================================================================
# Functions for dealing with label studio 
#==============================================================================
