#Install libraries
py -m pip install PyPDF2 pdfplumber tabula-py spaCy
py -m spacy download en_core_web_sm

#Load libraries 
import pdfplumber
import re
import spacy
import os
import pandas as pd

def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

#Define keywords and regular expressions


# Keywords related to the data of interest
keywords = [
    'aboveground biomass', 'belowground biomass', 'total biomass',
    'height', 'leaf area', 'LAI', 'germination', 'survival'
]

# Regular expressions to capture mean +/- SD or SE
mean_sd_pattern = re.compile(r'(\d+(\.\d+)?)(\s*±\s*|\s*\+\-\s*)(\d+(\.\d+)?)')
mean_se_pattern = re.compile(r'(\d+(\.\d+)?)(\s*\(\s*SE\s*=\s*)(\d+(\.\d+)?\s*\))')

# Function to extract relevant sentences and statistics
def extract_relevant_info(text):
    sentences = text.split('\n')
    relevant_info = []

    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in keywords):
            # Extract mean and SD/SE if present
            mean_sd_match = mean_sd_pattern.search(sentence)
            mean_se_match = mean_se_pattern.search(sentence)
            
            if mean_sd_match:
                mean = mean_sd_match.group(1)
                sd = mean_sd_match.group(4)
                relevant_info.append((sentence, mean, 'SD', sd))
            elif mean_se_match:
                mean = mean_se_match.group(1)
                se = mean_se_match.group(4)
                relevant_info.append((sentence, mean, 'SE', se))
            else:
                relevant_info.append((sentence, None, None, None))
    
    return relevant_info

#Load NLP to help parse out treatment effects 


nlp = spacy.load('en_core_web_sm')

def extract_treatment_effects(text):
    doc = nlp(text)
    treatment_effects = []
    
    for sent in doc.sents:
        sentence_text = sent.text.lower()
        if any(keyword in sentence_text for keyword in keywords):
            # Extract mean and SD/SE if present
            mean_sd_match = mean_sd_pattern.search(sentence_text)
            mean_se_match = mean_se_pattern.search(sentence_text)
            
            if mean_sd_match:
                mean = mean_sd_match.group(1)
                sd = mean_sd_match.group(4)
                treatment_effects.append((sentence_text, mean, 'SD', sd))
            elif mean_se_match:
                mean = mean_se_match.group(1)
                se = mean_se_match.group(4)
                treatment_effects.append((sentence_text, mean, 'SE', se))
            else:
                treatment_effects.append((sentence_text, None, None, None))
    
    return treatment_effects

    #Create a function to process multiple PDFs and compile the data into a single DataFrame.


def process_pdfs(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    all_data = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        text = extract_text(pdf_path)
        treatment_effects = extract_treatment_effects(text)
        for effect in treatment_effects:
            all_data.append({
                'PDF': pdf_file,
                'Sentence': effect[0],
                'Mean': effect[1],
                'Type': effect[2],
                'Value': effect[3]
            })
    
    return pd.DataFrame(all_data)

folder_path = './data/extraction/papers/'
all_data = process_pdfs(folder_path)
all_data.to_excel('all_extracted_data.xlsx', index=False)


sample_path = './data/extraction/papers/skarssonHeyser2015.pdf'
sample_text = extract_text(sample_path)
sample_relevant_info = extract_relevant_info(sample_text)
