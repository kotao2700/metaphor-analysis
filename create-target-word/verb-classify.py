import stanza
from stanza.models.common.doc import Document, Sentence, Word
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path
import csv
import pandas as pd

def main():
    v_freq = pd.read_csv('data/freqency/verb_freqency.csv')
    target_v_freq = v_freq[v_freq.freqency > 100000]
    target_words = target_v_freq.iloc[:,0]
    verb_classed = pd.DataFrame(0,columns=['auto','trans','is_collected'],index=target_words)
    nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse")

    def chunk_proc(chunk):
        doc = nlp("\n".join(chunk))
        for sentence in doc.sentences:
            for target_word in target_words:
                if target_word in sentence.text:
                    verb_class = classify_verbs(sentence,target_word)
                    if verb_class == 'trans':
                        verb_classed.at[target_word,'trans'] += 1
                    elif verb_class == 'auto':
                        verb_classed.at[target_word,'auto'] += 1
                    else:
                        pass
                    if verb_classed.at[target_word,'trans'] + verb_classed.at[target_word,'auto'] > 50:
                        verb_classed.at[target_word,'is_collected'] = 1

    def classify_verbs(sentence,target_word):
        words: List[Word] = sentence.words
        for word in words:
            if word.deprel == "obj" and words[word.head - 1].lemma == target_word:
                return 'trans'
        for word in words:
            if word.lemma == target_word:
                if word.pos == 'VERB':
                    return 'auto'
        return 'else'

    samples_of_chunk = 5000
    with open("/data/data/CC-100/en.txt") as f:
        chunk = []
        for i,line in tqdm(enumerate(f)):
            for target_word in target_words:
                if target_word in line:
                    if not verb_classed.at[target_word,'is_collected']:
                        chunk.append(line)
                        break
            if len(chunk) >= samples_of_chunk:
                chunk_proc(chunk)
                chunk = []
            if verb_classed.min().loc['is_collected'] == 1:
                break
        verb_classed.to_csv('./data/verb-classed.csv')

if __name__ == "__main__":
    main()


