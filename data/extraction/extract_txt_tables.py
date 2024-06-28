# Extract Text and Tables from PDF
import fitz  # PyMuPDF
import camelot

import spacy
import re

# Extract text from PDF
pdf_path = './papers/Semchenko2019.pdf'
pdf_document = fitz.open(pdf_path)
text = ""
for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)
    text += page.get_text()

# Extract tables from PDF
tables = camelot.read_pdf(pdf_path, pages='all')

# Load your trained NER model
nlp = spacy.load("custom_web_ner_abs_v381")

# Apply NER to the extracted text
doc = nlp(text)

# Extract treatment and response entities
treatments = []
responses = []
for ent in doc.ents:
    if ent.label_ == "TREATMENT":
        treatments.append(ent)
    elif ent.label_ == "RESPONSE":
        responses.append(ent)

#Use Dependency Parsing and Pattern Matching
# Function to find numerical value close to the response
def find_numerical_value(text, response):
    pattern = re.compile(r'(\d+(\.\d+)?\s?(%|g|kg|mg)?)')
    match = pattern.search(text)
    if match:
        return match.group()
    return None

# Extract numerical value for each response
response_data = []
for response in responses:
    numerical_value = find_numerical_value(text[response.start:response.end], response)
    response_data.append((response.text, numerical_value))

# Print extracted response data
print("Response Data:", response_data)

# Use dependency parsing to combine treatments
for token in doc:
    if token.dep_ in ("compound", "amod") and token.head.ent_type_ == "TREATMENT":
        combined_treatment = token.text + " " + token.head.text
        print("Combined Treatment:", combined_treatment)


# Print extracted treatments and responses
print("Treatments:", treatments)
print("Responses:", responses)

# Extract entities and associated numerical values
data = []
for ent in doc.ents:
    if ent.label_ in ["TREATMENT", "RESPONSE", "INOCTYPE"]:
        # Find numerical value in context
        pattern = re.compile(r'(\d+(\.\d+)?\s?(%|g|mg|kg)?)')
        match = pattern.search(text[ent.start_char-10:ent.end_char+10])
        if match:
            value = match.group()
        else:
            value = None
        data.append((ent.text, ent.label_, value))

# Convert to DataFrame and export to CSV
import pandas as pd

df = pd.DataFrame(data, columns=["Entity", "Label", "Value"])
df.to_csv("extracted_data.csv", index=False)

# Process tables and extract numerical values
for table in tables:
    df_table = table.df
    # Implement logic to map table columns to entity labels and numerical values
    # Example: df_table.loc[1, 'Biomass'] could be the biomass value for a treatment
