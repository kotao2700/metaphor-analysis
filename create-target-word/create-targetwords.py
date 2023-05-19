import csv
import pandas as pd

def main():
    default_target_words = ['kill','attack','break','eat','read','cut','heal','build','plant','ride']
    verb_classed = pd.read_csv('/home/kotaro/work/metaphor-analysis-m1/data/target-word/verb-classed.csv',index_col='verbs')
    target_words = []
    verb_threshold = 0.7
    verbs = verb_classed.index
    for verb in verbs:
        auto = verb_classed.at[verb,'auto']
        trans = verb_classed.at[verb,'trans']
        if trans > (auto + trans) * verb_threshold:
            target_words.append(verb)
    target_words = set(target_words + default_target_words)
    df = pd.DataFrame(target_words)
    df.to_csv('/home/kotaro/work/metaphor-analysis-m1/data/target-word/target-words.csv', index=False)
    print(len(target_words))

if __name__ == "__main__":
    main()
