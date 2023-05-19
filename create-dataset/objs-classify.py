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
from settings import TARGET_WORDS
import codecs

MODE = "obj"
# MODE = "verb+obj"
EX_DAY = '0512'
NUM_OF_OBJS = 10

@torch.no_grad()
def main():
    target_words = TARGET_WORDS
    nlp = en_core_web_lg.load()
    save_dir = {target_word:Path('./') for target_word in target_words}
    for target_word in target_words:
        save_dir[target_word] = Path(f"/home/kotaro/work/metaphor-analysis-m1/data/cc100/misnet-input-objs/misnet-input-{EX_DAY}/{target_word}")
        save_dir[target_word].mkdir(exist_ok=True, parents=True)

    objs: Dict[str,Dict] = {target_word:{} for target_word in target_words}

    for target_word in tqdm(target_words):
        if target_word in ['read', 'court', 'supply', 'heal', 'give', 'net', 'stake','keep', 'earn', 'cement', 'have']:
            continue
        with open(f'/home/kotaro/work/metaphor-analysis-m1/data/cc100/misnet-input/{target_word}.csv','r') as f:
            reader = csv.reader(f)
            for i,line in tqdm(enumerate(reader)):
                sentence = line[0].split(' ')
                obj_place = int(line.pop(-1))
                if '\u0000' in sentence or '\0' in sentence:
                    continue
                try:
                    objs[target_word][nlp(sentence[obj_place])[0].lemma_].append(line)
                except KeyError:
                    objs[target_word].update([(nlp(sentence[obj_place])[0].lemma_,[line])])
                if i > 1000000:
                    break
            for obj_dict in objs.values():
                for obj,sentence_list in obj_dict.items():
                    if len(sentence_list) > NUM_OF_OBJS and '/' not in obj:
                        with (save_dir[target_word] / f'{obj}.csv').open('a') as f:
                            writer = csv.writer(f)
                            for row in sentence_list:
                                writer.writerow(row)
        del objs[target_word]



if __name__ == "__main__":
    main()
