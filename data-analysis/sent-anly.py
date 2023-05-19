from settings import TARGET_WORDS
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer
import csv
from tqdm import tqdm
import pandas as pd

def main():
    target_words_sent = pd.DataFrame(0,columns=['avg','sum'],index=TARGET_WORDS)
    classifier = pipeline('sentiment-analysis')
    for target_word in tqdm(TARGET_WORDS):
        with open(f'/home/kotaro/work/metaphor-analysis-m1/data/cc100/misnet-input/{target_word}.csv','r') as f:
            reader = csv.reader(f)
            sentences = []
            for i,line in enumerate(reader):
                sentence = line[0]
                sentences.append(sentence)
                if i > 1000:
                    break
            sent_classed = classifier(sentences)
            verb_sent = 0
            for result in sent_classed:
                if result['label'] == 'POSITIVE':
                    verb_sent += result['score']
                elif result['label'] == 'NEGATIVE':
                    verb_sent -= result['score']
                else:
                    print('error')
            target_words_sent.loc[target_word,:] = [verb_sent/10000,verb_sent]
    target_words_sent.to_csv('/home/kotaro/work/metaphor-analysis-m1/data/result/verb_sentiment/verb_sentiment_0518.csv')

if __name__ == "__main__":
    main()