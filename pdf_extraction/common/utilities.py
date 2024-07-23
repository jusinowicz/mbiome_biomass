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


#==============================================================================
# Functions related to spacy  
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

#==============================================================================
# Functions for dealing with label studio 
#==============================================================================
