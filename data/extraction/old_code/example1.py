
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz

!pip install render

import scispacy
import spacy
import en_core_sci_sm
import en_core_sci_md

import en_ner_bc5cdr_md
from spacy import displacy
import pandas as pd



mtsample_df=pd.read_csv("mtsamples.csv")
print(mtsample_df.head())

#Test the models with sample data

# Pick specific transcription to use (row 3, column "transcription") and test the scispacy NER model
text = mtsample_df.loc[3, "transcription"]

text

nlp_sm = en_core_sci_sm.load()
doc = nlp_sm(text)

#Display resulting
displacy_image = displacy.render(doc, jupyter=True,style='ent')

# Entity
nlp_md = en_core_sci_md.load()
doc = nlp_md(text)

#Display resulting entity extraction

displacy_image = displacy.render(doc, jupyter=True,style='ent')

# disease, drug
nlp_bc = en_ner_bc5cdr_md.load()
doc = nlp_bc(text)
#Display resulting entity extraction
displacy_image = displacy.render(doc, jupyter=True,style='ent')


# ---------------------------
print("TEXT", "START", "END", "ENTITY TYPE")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)


mtsample_df.dropna(subset=['transcription'], inplace=True)
mtsample_df_subset = mtsample_df.sample(n=100, replace=False, random_state=42)

mtsample_df_subset.head()

# check drug

from spacy.matcher import Matcher

pattern = [{'ENT_TYPE':'CHEMICAL'}, {'LIKE_NUM': True}, {'IS_ASCII': True}]
matcher = Matcher(nlp_bc.vocab)
matcher.add("DRUG_DOSE", [pattern])

keep_result = []

# show DRUG_DOSE 
for transcription in mtsample_df_subset['transcription']:
    doc = nlp_bc(transcription)
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp_bc.vocab.strings[match_id]  # get string representation
        span = doc[start:end]  # the matched span adding drugs doses
        print(span.text, start, end, string_id,)

# show Disease
for ent in doc.ents:
  print( ent.text, ent.start_char, ent.end_char, ent.label_)