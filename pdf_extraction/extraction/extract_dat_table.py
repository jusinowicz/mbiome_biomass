#Install libraries
py -m pip install PyPDF2 pdfplumber tabula-py spaCy jpype1 PyMuPDF Pillow
py -m spacy download en_core_web_sm

#Load libraries 
import pdfplumber
import re
import spacy
import os
import tabula
import pandas as pd
import fitz  # PyMuPDF
import io
from PIL import Image

def extract_tables(pdf_path):
    # Extract all tables from the PDF and return as a list of DataFrames
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
    return tables

#Define keywords and regular expressions to find columns of interest


# Keywords related to the data of interest
keywords = [
    'aboveground biomass', 'belowground biomass', 'total biomass',
    'height', 'leaf area', 'LAI', 'germination', 'survival'
]

# Function to filter relevant columns based on keywords
def filter_relevant_columns(table):
    relevant_columns = []
    for col in table.columns:
        if any(keyword in col.lower() for keyword in keywords):
            relevant_columns.append(col)
    return table[relevant_columns] if relevant_columns else pd.DataFrame()


#Process all PDFs in the folder, extract relevant tables, and compile the data into a single DataFrame.
def process_pdfs(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    all_data = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        tables = extract_tables(pdf_path)
        
        for table in tables:
            filtered_table = filter_relevant_columns(table)
            if not filtered_table.empty:
                filtered_table['PDF'] = pdf_file  # Add PDF filename for reference
                # Use .loc to avoid SettingWithCopyWarning
                filtered_table = filtered_table.copy()
                filtered_table.loc[:, 'PDF'] = pdf_file
                all_data.append(filtered_table)
    
    # Combine all DataFrames into a single DataFrame
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
    else:
        combined_data = pd.DataFrame()  # Return an empty DataFrame if no data was found
    
    return combined_data

# Process all PDFs and create DataFrame
folder_path = './data/extraction/papers/'
all_data = process_pdfs(folder_path)
all_data.to_excel('all_extracted_data.xlsx', index=False)


sample_path = './data/extraction/papers/skarssonHeyser2015.pdf'
sample_tables = extract_tables(sample_path)
for idx, table in enumerate(sample_tables):
    filtered_table = filter_relevant_columns(table)
    if not filtered_table.empty:
        print(f"Filtered Table {idx}:")
        print(filtered_table)
