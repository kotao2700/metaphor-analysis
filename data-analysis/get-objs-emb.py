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
from settings import TARGET_WORDS
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
    model = Word2Vec.load(f'/home/kotaro/work/metaphor-analysis/data/w2v_model-{W2V_MODEL}/cc100_w2v-{W2V_MODEL}.model')

    save_dir = Path(f"/home/kotaro/work/metaphor-analysis-m1/data/result/w2v-{W2V_MODEL}-{MODE}-{EX_DAY}")
    save_dir.mkdir(exist_ok=True, parents=True)
    output_csv_load_dir = Path(f"/home/kotaro/work/metaphor-analysis-m1/data/cc100/misnet-output/misnet-output-{EX_DAY}")

    df = pd.DataFrame(0,columns=['var','metaphor-per','dist_score','cos_var'],index=target_words)


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

        target_embs = []
        target_objs = []
        usable_labels = []

        for target_obj,label_list in obj_dict.items():
            if MODE == "obj":
                try:
                    target_emb = model.wv[target_obj]
                except:
                    target_emb = None
            elif MODE == "verb+obj":
                target_verb_emb = model.wv[target_word]
                target_obj_emb = model.wv[target_obj]
                target_emb = torch.cat([target_verb_emb, target_obj_emb])

            if target_emb is not None:
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
                target_embs.append(target_emb)

        target_embs = np.array(target_embs)
        try:
            target_tsne = TSNE(n_components=2, learning_rate="auto", init="random",perplexity=50).fit_transform(target_embs)
        except:
            target_tsne = TSNE(n_components=2, learning_rate="auto", init="random",perplexity=10).fit_transform(target_embs)
        var = calc_var(target_embs)
        try:
            target_embs_random = random.sample(target_embs,200)
        except:
            target_embs_random = target_embs
        cos_var = cos_sim_var(target_embs_random)
        plt.figure(figsize=(8, 8))
        plt.title(f"{target_word},{var},{sum(usable_labels)/len(usable_labels)}")
        count = 0
        for (x, y), label, obj in zip(target_tsne, usable_labels, target_objs):
            if label:
                plt.plot(x, y, marker=".", c="red")
                plt.annotate(obj, (x, y), fontsize=8, c='red')
            else:
                plt.plot(x, y, marker=".", c="black")
                plt.annotate(obj, (x, y), fontsize=8,c='black')
            count += 1
            #if count > 200:
                #break
        sum_of_distance_rr = 0
        sum_of_distance_br = 0
        sum_of_distance_bb = 0
        rr_sum = 0
        br_sum = 0
        bb_sum = 0
        for target_emb,target_label in zip(target_embs,usable_labels):
            if target_label:
                for emb,label in zip(target_embs,usable_labels):
                    #distance = math.dist(target_emb,emb)
                    distance = cos_sim(target_emb,emb)
                    if label:
                        sum_of_distance_rr += distance
                        rr_sum += 1
                    else:
                        sum_of_distance_br += distance
                        br_sum += 1
            else:
                for emb,label in zip(target_embs,usable_labels):
                    #distance = math.dist(target_emb,emb)
                    distance = cos_sim(target_emb,emb)
                    if not label:
                        sum_of_distance_bb += distance
                        bb_sum += 1

        if not br_sum:
            br_sum = 1
        br_mean = sum_of_distance_br / br_sum
        rr_bb_mean = (sum_of_distance_bb + sum_of_distance_rr) / (bb_sum + rr_sum)
        if not br_mean:
            br_mean = 1
        #dist_score = rr_bb_mean / br_mean
        dist_score = br_mean / rr_bb_mean
        df.loc[target_word,:] = [var,sum(usable_labels)/len(usable_labels),dist_score,cos_var]
        try:
            plt.savefig(str(save_dir / f"{target_word}.png"))
        except:
            print('error')
        plt.close()
    df.to_csv(f'/home/kotaro/work/metaphor-analysis-m1/data/result/var-and-mp/var-and-mp-{EX_DAY}-{W2V_MODEL}.csv')

if __name__ == "__main__":
    main()
