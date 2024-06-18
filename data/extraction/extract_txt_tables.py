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
