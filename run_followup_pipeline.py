import os
import ast
import re

import pandas as pd

import prodigy
import spacy
from spacy.training import Corpus
from spacy.tokens import DocBin
from spacy.language import Language

from Weighted_TextCategorizer import TextCategorizer

MODEL_PATH = "/home/vs428/Documents/Moore/followup_model/model-best"
INPUT_FILE = "/home/vs428/project/Moore_data/Chest_CT_Reports_HVRADS(1).csv"
VERBOSITY = 10
GPU = True

if GPU:
    spacy.require_gpu()
    
def convert_dict_str_to_dict(x, col_name):
    tmp = ast.literal_eval(x[col_name])
    return tmp

nlp = spacy.load(MODEL_PATH)
data_in = pd.read_csv(INPUT_FILE)

if not "CT_text" in data_in.columns:
    raise KeyError("'CT_text' is not a column in your input data")

if not "ID" in data_in.columns:
    raise KeyError("'ID' is not a column in your input data")

_, ext = os.path.splitext(INPUT_FILE)
if ext[1:] not in ['csv', "tsv", "xlsx", "xls"]:
    raise ValueError(f"{ext} file type not supported")


# Convert text into IMPRESSIONS only
impressions = []
for idx, text in data_in.loc[~data_in['CT_text'].isnull(), 'CT_text'].items():
    try:
        impression = text
        if re.search(r"IMPRESSION:|Impression:", text):
            idx_start = re.search(r"IMPRESSION:|Impression:", text).start()
        else:
            idx_start = 0
        impression = text[idx_start:]
        impressions.append(impression)
    except ValueError:
        impressions.append(ann)
        
data_in.loc[~data_in['CT_text'].isnull(), 'CT_text_Impressions'] = impressions

if VERBOSITY:
    print("created impressions...", flush=True)
# taken from https://stackoverflow.com/a/44764557/1726404
'''
This works by using nlp.pipe and putting our records into tuples. We process it as tuples and get the context
In our work, the context is just the study id. 
We get the entity text, label, start and stop characters for each entity
we convert that to a json string, we then put the [context,json] together into a list
append this list to nlp_out
then turn nlp out into a df with 1 col being study id and the other being the nlp out
Finally we merge the df with our main data df. Now we have a column with the text
'''
import json
nlp_out = []
count = 0
for doc, ctx in nlp.pipe(list(data_in.loc[~data_in['CT_text_Impressions'].isnull(), 
                                          ['CT_text_Impressions', 'ID']].to_records(index=False)),
                                    as_tuples=True, batch_size=200, n_process=1):
    out_ = doc.cats
    nlp_out.append([ctx, json.dumps(out_, indent = 2)])
    
    if VERBOSITY:
        count +=1
        if count % 50 == 0 and VERBOSITY > 8:
            print(count, flush=True)
        elif count % 500 == 0 and VERBOSITY > 4:
            print(count, flush=True)
        elif count % 1000 == 0 and VERBOSITY > 2:
            print(count, flush=True)
        elif count % 5000 == 0 and VERBOSITY > 1:
            print(count, flush=True)

if VERBOSITY:
    print("ran predictions...", flush=True)
            
df = pd.DataFrame(nlp_out, columns=['ID', 'NLP_OUT'])
        
score_df = df.apply(convert_dict_str_to_dict, axis=1, col_name="NLP_OUT", result_type="expand")
df = pd.concat([df, score_df], axis=1)
df['y_pred'] = df[["NO_FOLLOWUP", "HARD_FOLLOWUP", "CONDITIONAL_FOLLOWUP"]].idxmax(axis=1)

x = data_in.merge(df, on="ID", how='left')
print(x['y_pred'].value_counts(normalize=True))

x = x.drop("NLP_OUT", axis = 1)

fname, ext = os.path.splitext(INPUT_FILE)

if ext[1:] == "tsv":
    x.to_csv(fname + "_predictions1" + ext, index=False, sep="\t")    
elif ext[1:] == "xls" or ext[1:] == "xlsx":
    x.to_excel(fname + "_predictions1" + ext, index=False)    
elif ext[1:] == "csv":
    x.to_csv(fname + "_predictions1" + ext, index=False)
else:
    x.to_csv(fname + "_predictions1.csv", index=False)
        