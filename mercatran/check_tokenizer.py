import logging
import os
import pathlib
import sys
from argparse import ArgumentParser
from functools import partial
import pandas as pd

import config
import torch
import torch.nn as nn
from data import (
    MASK_TOKEN,
    UserItemInteractionDataset,
    collate_batch_item,
    collate_batch_item_val,
    sequence_dataset,
    train_tokenizer, collate_batch_item_wrapper, collate_batch_item_val_wrapper,
)
from embed import (
    ItemEmbeddings,
    PositionalEncoding,
    UserEmbeddings,
    create_item_encoder_mask,
    create_user_target_mask,
)
from eval_utils import Evaluator
from model import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    ThreeTower,
    model_initialization,
    rate,
)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def main(args):
    model_dir = pathlib.Path(args.save_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_path
    data = pd.read_pickle(data_path)
    seq_dataset = pd.DataFrame.from_dict(data, orient='index')
    seq_dataset = seq_dataset[
        ['seq_user_id', 'name', 'category_name', 'brand_name', 'category_id', 'brand_id', 'item_id', 'event_id']]
    tokenizer = train_tokenizer(df_train=seq_dataset)
    print("Mask token ID:")
    print(tokenizer.token_to_id(MASK_TOKEN))



def parse_args(description):

    parser = ArgumentParser(description=description)
    parser.add_argument('--save_path', type=str, default='./checkpoint/')
    parser.add_argument('--metrics_path', type=str, default='./metrics/')
    parser.add_argument('--tokenizer_save_name', type=str,
                        default='tokenizer.json')
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--max_len", type=int, default=22)
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--test_frac", type=int, default=0.3)
    parser.add_argument("--use_event_id", action='store_true',
                        help="Use event_id in the dataset")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='\n%(message)s')
    sys.exit(main(parse_args("Check tokenizer in pipeline.")))
