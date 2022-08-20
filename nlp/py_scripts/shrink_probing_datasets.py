import pandas as pd
import numpy as np
import os

tasks = ["bigram_shift", "coordination_inversion", "obj_number", "odd_man_out", "past_present", "sentence_length", "subj_number", "top_constituents", "tree_depth"] 
data_path = "../SentEval/data/probing_large/"
output_path = "../SentEval/data/probing/"

if not os.path.exists(output_path):
    os.mkdir(output_path)

train_size = 10000 # 100
val_size = 1000 #10
test_size = 1000 #10
seed = 42

for task in tasks:
    f = open(data_path + task + ".txt", encoding="utf8")
    f_text = f.read()
    f_lines = f_text.splitlines()
    df = pd.DataFrame([line.split('\t') for line in f_lines], columns=["type", "label", "sentence"])
    df_tr = df[df["type"]=="tr"].sample(n=train_size, random_state=seed)   
    df_va = df[df["type"]=="va"].sample(n=val_size, random_state=seed)
    df_te = df[df["type"]=="te"].sample(n=test_size, random_state=seed) 
    small_df = pd.concat([df_tr, df_va, df_te], ignore_index=True)
    small_df.to_csv(f"{output_path}{task}.txt", sep='\t', header=None, index=None)

