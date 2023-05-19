import pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import glob
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from pathlib import Path
import os
import os.path as osp
import sys
import argparse
from transformers import AdamW, get_linear_schedule_with_warmup
import pickle
from model import Model
from data_loader import dataset, collate_fn
from data_process import VUA_All_Processor
from utils import Logger
from configs.default import get_config
from train_val import val, load_plm, set_random_seeds
import csv

EX_DAY = '0512'

TARGET_WORDS = ['read', 'court', 'supply', 'heal', 'give', 'net', 'stake',\
    'keep', 'earn', 'cement', 'pocket', 'loan', 'eat', 'pull', 'excuse',\
    'spell', 'distance', 'join', 'ease', 'milk', 'express', 'pick', 'influence',\
    'make', 'tell', 'view', 'kiss', 'attack', 'plant', 'welcome', 'watch', 'harm',\
    'meet', 'ride', 'find', 'gain', 'kill', 'carry', 'voice', 'cross', 'hand', 'free',\
    'cut', 'hold', 'waste', 'send', 'lose', 'raid', 'cause', 'put', 'cost', 'exchange',\
    'take', 'witness', 'shed', 'interest', 'piece', 'break', 'save', 'track', 'build',\
    'raise', 'allow', 'lift', 'lack', 'teach', 'buy']


def parse_option():
    parser = argparse.ArgumentParser(description='Train on VUA All dataset')
    parser.add_argument('--cfg', type=str, default='./configs/vua_all.yaml', metavar="FILE",
                        help='path to config file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--log', default='vua_all', type=str)
    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config

def main(args):
    output_dir = Path(f"/home/kotaro/work/metaphor-analysis-m1/data/cc100/misnet-output/misnet-output-{EX_DAY}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.stdout = Logger(osp.join(args.TRAIN.output, f'{args.log}.txt'))
    print(args)
    set_random_seeds(args.seed)

    # prepare train-val datasets and dataloaders
    processor = VUA_All_Processor(args)

    # load model
    plm = load_plm(args)
    model = Model(args=args, plm=plm)
    model.cuda()
    done_words = ['read', 'court', 'supply', 'heal', 'give', 'net', 'stake',\
    'keep', 'earn', 'cement', 'pocket', 'loan', 'eat', 'pull', 'excuse',\
    'spell', 'distance', 'join', 'ease', 'milk', 'express', 'pick', 'influence',\
    'make']

    print("Evaluate only")
    model.load_state_dict(torch.load('./checkpoints/best_vua_all.pth'))
    for target_word in TARGET_WORDS:
        if target_word in done_words:
            continue
        files = glob.glob(f"/home/kotaro/work/metaphor-analysis-m1/data/cc100/misnet-input-objs/misnet-input-{EX_DAY}/{target_word}/*")
        for file in files:
            obj = file.split("/")[-1].split(".")[0]
            my_data = processor.my_get_examples(file)
            my_set = dataset(my_data)
            my_data_loader = DataLoader(my_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)
            preds = val(model, my_data_loader)
            print(preds)
            with open(f'/home/kotaro/work/metaphor-analysis-m1/data/misnet-result-{EX_DAY}.pkl',mode='wb') as f:
                pickle.dump(preds,f)
            if preds == []:
                continue
            with (output_dir / f"{target_word}.csv").open("a") as f:
                writer = csv.writer(f)
                list = [obj]
                list.extend(preds)
                writer.writerow(list)



if __name__ == '__main__':
    args = parse_option()
    main(args)
