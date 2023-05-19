import pandas as pd
import numpy as np
TARGET_WORDS = TARGET_WORDS = ['read', 'court', 'supply', 'heal', 'give', 'net', 'stake',\
    'keep', 'earn', 'cement', 'pocket', 'loan', 'eat', 'pull', 'excuse',\
    'spell', 'distance', 'join', 'ease', 'milk', 'express', 'pick', 'influence',\
    'make', 'tell', 'view', 'kiss', 'attack', 'plant', 'welcome', 'watch', 'harm',\
    'meet', 'ride', 'find', 'gain', 'kill', 'carry', 'voice', 'cross', 'hand', 'free',\
    'cut', 'hold', 'waste', 'send', 'lose', 'raid', 'cause', 'put', 'cost', 'exchange',\
]

def main():
    mrc_data = pd.read_csv('/home/kotaro/work/metaphor-analysis-m1/data/mrc/mrc_dataset/mrc2_target_words.csv',index_col=0)
    var_mp_data = pd.read_csv('/home/kotaro/work/metaphor-analysis-m1/data/result/var-and-mp/mp-0512-skipgram.csv',index_col=0)
    target_words = TARGET_WORDS
    target_mrc_data = mrc_data.loc[target_words,:]
    print(var_mp_data.loc[target_words,:])
    concat_data = pd.concat([target_mrc_data,var_mp_data.loc[target_words,:]],axis=1,join='inner')
    concat_data.to_csv('/home/kotaro/work/metaphor-analysis-m1/data/result/var-and-mp/concat_data-0512.csv')
    concat_data = concat_data.sort_values('metaphor-per')
    concat_data.to_latex('uoo.tex')
    fam_list = np.array(concat_data.loc[:,'fam'])
    conc_list = np.array(concat_data.loc[:,'conc'])
    img_list = np.array(concat_data.loc[:,'img'])
    aoa_list = np.array(concat_data.loc[:,'aoa'])
    mp_list = np.array(concat_data.loc[:,'metaphor-per'])
    corr = np.corrcoef([mp_list,fam_list,conc_list,img_list])
    print(corr)
    new_mp_list = []
    new_aoa_list = []
    for (aoa,mp) in zip(aoa_list,mp_list):
        if aoa == 0:
            continue
        else:
            new_aoa_list.append(aoa)
            new_mp_list.append(mp)
    print(len(new_aoa_list))
    print(np.corrcoef([np.array(new_mp_list),np.array(new_aoa_list)]))


if __name__ == "__main__":
    main()

