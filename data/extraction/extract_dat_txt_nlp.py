#==============================================================================
# Attempt to automate extraction of treatment effects from scientific papers, 
# along with covariates or other descriptors of interes. 
#
# Install libraries
# One note on installation. This package, which needs to be installed for NLP: 
# py -3.10 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz
## py -3.10 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_scibert-0.5.0.tar.gz
#
# was very finicky. I could only get both it and spacy to install and run 
# on python 3.10, not the current 3.12. It seemed possible maybe with 3.11 but 
# very finicky to set up. 
# My recommendation is to install this first and let it install its own 
# version of spacy and dependencies (something with pydantic versions seems
# to be the problem).
# 
# The package en_core_sci_md-0.4.0 also requires that C++ is installed on your system, so visual studio build
# tools on Windows.
#
# py -m pip install PyPDF2 pdfplumber tabula-py jpype1 PyMuPDF Pillow nltk
# For spacy:
# py -m spacy download en_core_web_sm
# For NER: 
#
# py -3.10 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz
# py -3.10 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_craft_md-0.5.0.tar.gz

#==============================================================================
py -3.10

#PDF extraction
import fitz  # PyMuPDF
#Text preprocessing
import re
import nltk
from nltk.tokenize import sent_tokenize
#NER and NLP
import spacy
# Load pre-trained SciBERT model from spaCy
nlp = spacy.load("en_core_sci_md")
#nlp = spacy.load("en_core_sci_scibert")
#nlp_bc = spacy.load("en_ner_bc5cdr_md")
nlp_bc = spacy.load("en_ner_craft_md")
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

def extract_entities(text):
	doc = nlp(text)
	entities = [(ent.text, ent.label_) for ent in doc.ents]
	return entities

# Apply NER to the Methods section to identify treatments and covariates
methods_text = " ".join(sections.get('methods', []))
entities = extract_entities(methods_text)
print(entities)

#Step 5: Pattern Matching and Keyword Extraction
def extract_keywords(text, keywords):
	keyword_sentences = []
	for keyword in keywords:
		matches = re.findall(rf'\b{keyword}\b', text, re.I)
		keyword_sentences.extend(matches)
	return keyword_sentences

treatment_keywords = ['treatment', 'therapy', 'drug', 'intervention']
response_keywords = ['response', 'outcome', 'result', 'effect']
covariate_keywords = ['covariate', 'variable', 'factor']

treatment_sentences = extract_keywords(methods_text, treatment_keywords)
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