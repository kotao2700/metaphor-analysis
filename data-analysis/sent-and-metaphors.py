import csv
from transformers import pipeline
import matplotlib.pyplot as plt
import pandas as pd
from google.cloud import language_v1
EX_DAY = '0531'

def analyze_sentiment(content):

    client = language_v1.LanguageServiceClient()

    # content = 'Your text to analyze, e.g. Hello, world!'

    if isinstance(content, bytes):
        content = content.decode("utf-8")

    type_ = language_v1.Document.Type.PLAIN_TEXT
    document = {"type_": type_, "content": content}

    response = client.analyze_sentiment(request={"document": document})
    sentiment = response.document_sentiment
    return sentiment.score

def main():
    metaphor_sentences = []
    else_sentences = []
    all_sentences = [metaphor_sentences,else_sentences]
    all_results = {'metaphors':[],'elses':[]}
    with open(f'/home/kotaro/work/metaphor-analysis-m1/data/cc100/misnet-output/sentiment/misnet-output-sent-{EX_DAY}/misnet-output-sent-{EX_DAY}.csv','r') as f:
        reader = csv.reader(f)
        print(reader)
        for row in reader:
            with open(f'/home/kotaro/work/metaphor-analysis-m1/data/cc100/misnet-input-sent/misnet-input-sent-{EX_DAY}.csv','r') as f:
                count = 0
                for line,label in zip(f,row):
                    if int(label):
                        metaphor_sentences.append(line)
                    else:
                        else_sentences.append(line)
        for i,sentences in enumerate(all_sentences):
            for sentence in sentences:
                score = analyze_sentiment(sentence)
                if i == 0:
                    all_results['metaphors'].append(score)
                else:
                    all_results['elses'].append(score)
        df = pd.DataFrame.from_dict(all_results,orient="index")
        df.to_csv('/home/kotaro/work/metaphor-analysis-m1/data/result/sent-result.csv')
        plt.hist(all_results['metaphors'])
        plt.savefig('/home/kotaro/work/metaphor-analysis-m1/data/result/metaphor-hist.png')
        plt.hist(all_results['elses'])
        plt.savefig('/home/kotaro/work/metaphor-analysis-m1/data/result/elses-hist.png')
if __name__ == '__main__':
    main()