from pathlib import Path
import torch
import glob
import stanza
from gensim.models import word2vec,Word2Vec
import pandas as pd
from tqdm import tqdm
import csv
from typing import Dict
import en_core_web_lg
import collections
from scipy.stats import entropy
from settings import TARGET_WORDS

def main():
    nlp = en_core_web_lg.load()
    target_words = TARGET_WORDS
    save_dir = Path(f"/home/kotaro/work/metaphor-analysis-m1/data/result/var-and-mp")
    save_dir.mkdir(exist_ok=True, parents=True)
    input_dir = Path(f"/home/kotaro/work/metaphor-analysis-m1/data/cc100/misnet-input")

    df = pd.DataFrame(0,columns=['KoO','entropy'],index=target_words)

    for target_word in tqdm(target_words):
        f = open(input_dir / f'{target_word}.csv','r')
        objs = []
        reader = csv.reader(f)
        for i,line in tqdm(enumerate(reader)):
            if '\u0000' in line:
                continue
            obj_place = int(line.pop(-1))
            sentence = line[0].split(' ')
            obj = sentence[int(obj_place)]
            obj = nlp(obj)[0].lemma_
            objs.append(obj)
            if i > 10000:
                break
        objs_set = set(objs)
        prob_dist = []
        objs_counter = collections.Counter(objs)
        for o in objs_set:
            prob_dist.append(objs_counter[o]/len(objs))
        obj_entropy = entropy(prob_dist,base=2)
        kind_of_objs = len(objs_set)
        df.loc[target_word,:] = [kind_of_objs,obj_entropy]
        f.close()
    df.to_csv(save_dir / 'KoO-0510.csv')

if __name__ == "__main__":
    main()

