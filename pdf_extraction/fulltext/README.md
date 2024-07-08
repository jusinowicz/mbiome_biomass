# Soil microbiomes and biomass meta analysis project
## This is the full text parsing and phase 1 of data extraction
1. **meta_analyze_get_full.py**: Using the retrieved citations based on keyword search via PubMed, try to pull full texts from open access sources. 

- Automate creation (if needed) and uploading of Methods and Results sections from the full text to the labeling project in Label Studio (currently ). Although I am not currently pursuing this, this could be used to train a second custom NER for the full text.

- Use the current version of the NER to streamline labeling by generating predictions. The labeling process is iterative! Label, predict, correct, generate a new version of the NER (via meta_analyze_model_update.py), use it to label, predict...

2. **extract_responses_txt_v2.py**: This is Step 1 in the full text parsing pipeling. Use NLP and a custom NER model to extract the TREATMENTs and RESPONSEs from the text of the Methods and Results sections in scientfic papers. Here, we are trying to glean info from the text before trying more complex table-extraction and figure-extractionmethods.

- The final output is a df/CSV with columns STUDY, TREATMENT, RESPONSE, CARDINAL, PERCENTAGE, SENTENCE, ISTABLE. There are separate columns for CARDINAL (a numeric) response) and PERCENTAGE because the NER recognizes them separately. This is useful because it helps determine whether actual units of biomass response are being identified or the ratio of response (percentage). 

- SENTENCE is the sentence that was parsed for the information in the table ISTABLE is a numeric ranking from 0-2 which suggests how likely it is that the the parsed information came from a table that the pdf-parsing grabbed. In this case, the results are most definitely not to be trusted. 

- This table is meant to help determine what information is available in the paper and indicate whether further downstream extraction is necessary. 

