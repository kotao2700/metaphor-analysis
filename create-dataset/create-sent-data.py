from pathlib import Path
import torch
import glob
import stanza
from gensim.models import word2vec,Word2Vec
import pandas as pd
from tqdm import tqdm
import csv
from typing import Dict
from settings import TARGET_WORDS
import codecs

MODE = "obj"
# MODE = "verb+obj"
EX_DAY = '0531'
NUM_OF_OBJS = 10

@torch.no_grad()
def main():
    target_words = TARGET_WORDS
    save_dir = Path('/home/kotaro/work/metaphor-analysis-m1/data/cc100/misnet-input-sent')
    save_dir.mkdir(exist_ok=True, parents=True)
    row = []

    for target_word in tqdm(target_words):
        if target_word in ['have']:
            continue
        with open(f'/home/kotaro/work/metaphor-analysis-m1/data/cc100/misnet-input/{target_word}.csv','r') as f:
            reader = csv.reader(f)
            for i,line in tqdm(enumerate(reader)):
                sentence = line[0].split(' ')
                obj_place = int(line.pop(-1))
                if '\u0000' in sentence or '\0' in sentence:
                    continue
                row.append(line)
                if i > 80:
                    break
    with (Path(save_dir / f'misnet-input-sent-{EX_DAY}.csv')).open('a') as f:
        writer = csv.writer(f)
        for line in row:
            writer.writerow(line)



if __name__ == "__main__":
    main()
