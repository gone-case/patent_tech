import os
import sys
import pandas as pd
import numpy as np

from tqdm import tqdm

# file paths (fill these in before running)
worker1_path = ""
worker2_path = ""
worker3_path = ""
sheet_name = ""

if not all([worker1_path, worker2_path, worker3_path]) or not all(map(os.path.exists, [worker1_path, worker2_path, worker3_path])):
    print("\n[Notice] Please specify valid Excel file paths for `worker1_path`, `worker2_path`, and `worker3_path` in `data_processing.py`.\n")
    sys.exit(0)

# datas
worker1 = pd.read_excel(worker1_path, sheet_name=sheet_name if sheet_name else None)
worker2 = pd.read_excel(worker2_path, sheet_name=sheet_name if sheet_name else None)
worker3 = pd.read_excel(worker3_path, sheet_name=sheet_name if sheet_name else None)

# calculates values
sumr = worker1['similarity'] + worker2['similarity'] + worker3['similarity']
average = sumr / 3
distance = ((average - worker1['similarity']) ** 2) + ((average - worker2['similarity']) ** 2) + ((average - worker3['similarity']) ** 2)
threshold = 8

expert_ratings = []

for r, a in zip(distance, average):
    if r >= threshold:
        expert_ratings.append(-100)
    else:
        expert_ratings.append(a)

# eport results divide data needing expert rating & rights rating
worker1['similarity'] = expert_ratings

expert = worker1[worker1['similarity'] == -100]
worker1 = worker1[worker1['similarity'] != -100]
worker1.to_excel('processed_worker1.xlsx', index=False)
expert.to_excel('expert_ratings.xlsx', index=False)
print("Saved processed data to processed_worker1.xlsx and expert_ratings.xlsx")