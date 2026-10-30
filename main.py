from train_models.sbert import Sentence_Bert
from train_models.sim_lstm import Siamese_LSTM_Model
from train_models.sim_bilstm import Siamese_BiLSTM_Model
from train_models.sim_cnn import Siamese_CNN_Model
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import os
import sys

import torch
import torch.nn as nn
import pandas as pd
import ast

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# bert
sbert = Sentence_Bert(device=device).eval()

# paths to trained weights and dataset (fill these in before running)
# paths to trained weights and dataset (fill these in before running)
lstm_weights_path = ""
bilstm_weights_path = ""
rawcnn_weights_path = ""
data_path = "patent_dataset.xlsx"
output_path = "output_result.xlsx"

# check if paths are configured
if not data_path or not os.path.exists(data_path):
    print("\n[Notice] Please specify the path to your dataset Excel file in `main.py` (variable `data_path`).")
    print("Example: data_path = 'patent_dataset.xlsx'\n")
    sys.exit(0)

# other model
lstm = Siamese_LSTM_Model().eval()
bilstm = Siamese_BiLSTM_Model().eval()
rawcnn = Siamese_CNN_Model().eval()

if lstm_weights_path and os.path.exists(lstm_weights_path):
    lstm.load_state_dict(torch.load(lstm_weights_path, map_location=device))
else:
    print("[Info] No custom LSTM weights provided; using initialized baseline model.")

if bilstm_weights_path and os.path.exists(bilstm_weights_path):
    bilstm.load_state_dict(torch.load(bilstm_weights_path, map_location=device))
else:
    print("[Info] No custom BiLSTM weights provided; using initialized baseline model.")

if rawcnn_weights_path and os.path.exists(rawcnn_weights_path):
    rawcnn.load_state_dict(torch.load(rawcnn_weights_path, map_location=device))
else:
    print("[Info] No custom CNN weights provided; using initialized baseline model.")

# data prepare
data = pd.read_excel(data_path, engine="openpyxl")

A_codes = list(map(lambda x : ast.literal_eval(x), data['code']))
B_codes = list(map(lambda x : ast.literal_eval(x), data['target_ipc']))

A_titles_abstracts = data['meta'].tolist()
B_titles_abstracts = data['target_meta'].tolist()


def get_sbert_similarity(A_codes = None, B_codes = None, A_titles_abstracts = None, B_titles_abstracts = None, method : str = "proposed"): # method = ["proposed", "SD", "TD"]
    similarities = []
    for a_code, b_code, a_title_abstract, b_title_abstract in tqdm(zip(A_codes, B_codes, A_titles_abstracts, B_titles_abstracts), total = len(A_codes), desc = "calculating sbert similarity"):
        tile_abstract = {"documents_0" : [a_title_abstract], "documents_1" : [b_title_abstract]}
        ipc_document = {"documents_0" : [a_code], "documents_1" : [b_code]}
        if method == "proposed":
            similarity = sbert.get_similarity(tile_abstract, ipc_document)
        elif method == "SD":
            similarity = sbert.get_document_similarity(tile_abstract)
        elif method == "TD":
            similarity = sbert.ipc_similarity(ipc_document)
        similarities.append(similarity.item())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return similarities



def get_other_similaritiy(model): # model = ["lstm", "bilstm", "cnn"]
    model = model.to(device)
    similarities = []
    for a_title_abstract, b_title_abstract in tqdm(zip(A_titles_abstracts, B_titles_abstracts), total = len(A_titles_abstracts), desc = "calculating similarity"):
        a_title_abstract = sbert.tokenizer(a_title_abstract, return_tensors = 'pt', padding = 'max_length', is_split_into_words = False, add_special_tokens = True, truncation = True, max_length = 512).to(device)
        b_title_abstract = sbert.tokenizer(b_title_abstract, return_tensors = 'pt', padding = 'max_length', is_split_into_words = False, add_special_tokens = True, truncation = True, max_length = 512).to(device)
        a_title_abstract, b_title_abstract = a_title_abstract.input_ids, b_title_abstract.input_ids
        similarity = model(a_title_abstract, b_title_abstract).item()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        similarities.append(similarity)
    return similarities


data['proposed'] = get_sbert_similarity(A_codes, B_codes, A_titles_abstracts, B_titles_abstracts, "proposed")
data['Bert'] = get_sbert_similarity(A_codes, B_codes, A_titles_abstracts, B_titles_abstracts, "SD")

data['lstm_similarity'] = get_other_similaritiy(lstm)
data['bilstm_similarity'] = get_other_similaritiy(bilstm)
data['cnn_similarity'] = get_other_similaritiy(rawcnn)

comparisons = ['proposed', 'Bert', 'lstm_similarity', 'bilstm_similarity', 'cnn_similarity']
results = {}
pvalues = {}

for key in comparisons:
    results[f'{key}_pearson'] = pearsonr(np.array(data[key]), np.array(data['similarity'])).statistic
    results[f'{key}_spearman'] = spearmanr(np.array(data[key]), np.array(data['similarity'])).statistic

    pvalues[f'{key}_pearson'] = pearsonr(np.array(data[key]), np.array(data['similarity'])).pvalue
    pvalues[f'{key}_spearman'] = spearmanr(np.array(data[key]), np.array(data['similarity'])).pvalue

experiment_result = pd.DataFrame({'model' : results.keys(), 'cor' : results.values()})
experiment_result_pvalue = pd.DataFrame({'model' : pvalues.keys(), 'cor' : pvalues.values()})

output_data_path = 'output_data.xlsx'
output_pvalue_path = 'output_pvalues.xlsx'
output_result_path = 'output_results.xlsx'

data.to_excel(output_data_path, index = False)
experiment_result_pvalue.to_excel(output_pvalue_path, index = False)
experiment_result.to_excel(output_result_path, index = False)
print(f"Results saved to {output_result_path}, {output_pvalue_path}, {output_data_path}")
