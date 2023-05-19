from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pickle
from typing import List, Tuple
from stanza.models.common.doc import Document, Sentence, Word
import torch
from gensim.models import word2vec,Word2Vec
import pandas as pd
from tqdm import tqdm
import csv
import random
TARGET_WORDS = ['read', 'court', 'supply', 'heal', 'give', 'net', 'stake',\
    'keep', 'earn', 'cement', 'pocket', 'loan', 'eat', 'pull', 'excuse',\
    'spell', 'distance', 'join', 'ease', 'milk', 'express', 'pick', 'influence',\
    'make', 'tell', 'view', 'kiss', 'attack', 'plant', 'welcome', 'watch', 'harm',\
    'meet', 'ride', 'find', 'gain', 'kill', 'carry', 'voice', 'cross', 'hand', 'free',\
    'cut', 'hold', 'waste', 'send', 'lose', 'raid', 'cause', 'put', 'cost', 'exchange',\
]
EX_DAY = '0512'

MODE = "obj"
# MODE = "verb+obj"
W2V_MODEL = 'skipgram'
#W2V_MODEL = 'cbow'

def cos_sim(x,y):
    x = np.array(x)
    y = np.array(y)
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

def cos_sim_var(embs):
    var = 0
    count = 0
    for emb1 in embs:
        for emb2 in embs:
            var += cos_sim(emb1,emb2)
            count += 1
    return (var / count)

def calc_var(embs):
    normalized_embs = []
    var = 0
    for emb in embs:
        norm_emb = emb / np.linalg.norm(emb)
        normalized_embs.append(norm_emb)
    for dim in np.array(normalized_embs).T:
        var = var + np.var(dim)
    return var

@torch.no_grad()
def main():
    target_words = TARGET_WORDS

    save_dir = Path(f"/home/kotaro/work/metaphor-analysis-m1/data/result/w2v-{W2V_MODEL}-{MODE}-{EX_DAY}")
    save_dir.mkdir(exist_ok=True, parents=True)
    output_csv_load_dir = Path(f"/home/kotaro/work/metaphor-analysis-m1/data/cc100/misnet-output/misnet-output-{EX_DAY}")

    df = pd.DataFrame(0,columns=['metaphor-per'],index=target_words)


    for target_word in tqdm(target_words):
        if target_word == 'read':
            continue
        if target_word in TARGET_WORDS:
            with open(output_csv_load_dir / f'{target_word}.csv','r') as f:
                reader = csv.reader(f)
                melbert_output = [row for row in reader]
        else:
            with open(added_output_csv_load_dir / f'{target_word}.csv','r') as f:
                reader = csv.reader(f)
                melbert_output = [row for row in reader]
        obj_dict = {obj[0]:list(map(int,obj[1:])) for obj in melbert_output}
        target_objs = []
        usable_labels = []

        for target_obj,label_list in obj_dict.items():
            label = 0
            par = sum(label_list) / len(label_list)
            if par > 0.7:
                label = 1
            elif par < 0.3:
                label = 0
            else:
                continue
            target_objs.append(target_obj)
            usable_labels.append(label)
        df.loc[target_word,:] = [sum(usable_labels)/len(usable_labels)]
    df.to_csv(f'/home/kotaro/work/metaphor-analysis-m1/data/result/var-and-mp/mp-{EX_DAY}-{W2V_MODEL}.csv')

if __name__ == "__main__":
    main()
