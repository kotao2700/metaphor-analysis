import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from settings import TARGET_WORDS


def main():
    target_words = TARGET_WORDS
    #mrc_data = pd.read_csv('./data/mrc/mrc_dataset/mrc2_target_words.csv',index_col=0)
    #target_mrc_data = mrc_data.loc[target_words,:]
    KoO_data = pd.read_csv("/home/kotaro/work/metaphor-analysis-m1/data/result/var-and-mp/KoO-0510.csv",index_col=0)
    freq_data = pd.read_csv('/home/kotaro/work/metaphor-analysis-m1/data/freqency/verb_freqency.csv',index_col=0)
    var_mp_data = pd.read_csv('/home/kotaro/work/metaphor-analysis-m1/data/result/var-and-mp/var-and-mp-0509-skipgram.csv',index_col=0)
    target_freq_data = freq_data.loc[target_words,:]
    target_KoO_data = KoO_data.loc[target_words,:]
    target_vm_data = var_mp_data.loc[target_words,:]
    concat_data = pd.concat([target_freq_data,target_vm_data,target_KoO_data],axis=1)
    concat_data.to_csv('/home/kotaro/work/metaphor-analysis-m1/data/result/var-and-mp/concat_data-all-0202.csv')
    concat_sorted_data = concat_data.sort_values('metaphor-per')
    freq_list = scipy.stats.zscore(np.array(concat_data.loc[:,'freqency']))
    var_list = np.array(concat_data.loc[:,'var'])
    entropy_list = np.array(concat_data.loc[:,'entropy'])
    KoO_list = np.array(concat_data.loc[:,'KoO'])
    mp_list = np.array(concat_data.loc[:,'metaphor-per'])
    cos_var_list = np.array(concat_data.loc[:,'cos_var'])
    dist_score_list = np.array(concat_data.loc[:,'dist_score'])
    plt.hist(dist_score_list,range=(3.0,5.0),ec='black')
    norm_concat_data = concat_data
    norm_concat_data['norm_var'] = scipy.stats.zscore(var_list)
    norm_concat_data['norm_en'] = scipy.stats.zscore(entropy_list)
    norm_concat_data['norm_KoO'] = scipy.stats.zscore(KoO_list)
    norm_concat_data['norm_cos_var'] = scipy.stats.zscore(cos_var_list)
    norm_concat_data.loc[TARGET_WORDS].sort_values('metaphor-per').to_latex('./test_latex.tex')
    plt.yticks(np.arange(0, 12, 2))
    for score in dist_score_list:
        if score < 0.6 or score > 1.0:
            print(score)
    print(len(dist_score_list))
    plt.savefig('/home/kotaro/work/metaphor-analysis-m1/data/result/dist_score_hist2.png')
    corr = np.corrcoef([mp_list,freq_list,KoO_list,entropy_list,cos_var_list,var_list])
    print(corr)

if __name__ == "__main__":
    main()