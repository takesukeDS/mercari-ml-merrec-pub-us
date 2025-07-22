import logging
import os
import pathlib
import sys
from argparse import ArgumentParser
from functools import partial
from tokenizers import Tokenizer

import config
import torch
import torch.nn as nn
from data import (
    MASK_TOKEN,
    UserItemInteractionDataset,
    collate_batch_item,
    collate_batch_item_val,
    sequence_dataset,
    train_tokenizer,
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
import pandas as pd


EVENT_ID_TO_TOKEN = {
    "item_view": "AAitem_viewAA",
    "item_like":  "AAitem_likeAA",
    "item_add_to_cart_tap":  "AAitem_add_to_cart_tapAA",
    "offer_make":  "AAoffer_makeAA",
    "buy_start":  "AAbuy_startAA",
    "buy_comp":  "AAbuy_compAA",
}

SHIPPER_ID_TO_TOKEN = {
    1: "BBBuyerBB",
    2: "BBSellerBB"
}


def combine_tokens(tokens, trim=True):
    if isinstance(tokens, pd.Series):
        tokens = tokens.tolist()
    if trim:
        # Remove the first and last characters (e.g., '<' and '>')
        tokens = [token[1:-1] for token in tokens]
    combined = ''.join(tokens)
    return "CC" + combined + "CC"

def all_hierarchical_category(lst):
    return [
        ' '.join([part for part in s.split('#')[::-1] if part.strip() != ''])
        for s in lst
    ]

def min_hierarchical_category(lst):
    result = []
    for s in lst:
        for part in s.split('#'):
            if part.strip():
                result.append(float(part))
                break  # その文字列については先頭のみでOK
    return result

dict_event = {
    'item_view': 'aaaviewbbb',
    'item_like': 'aaalikebbb',
    'item_add_to_cart_tap': 'aaaaddbbb',
    'buy_start': 'aaastartbbb',
    'offer_make': 'aaaofferbbb',
    'buy_comp': 'aaacompbbb'
}

def calculate_time_diffs(timestamp_seq):
    dt_series = pd.to_datetime(pd.Series(timestamp_seq), format='%Y-%m-%d %H:%M:%S.%f')
    time_deltas = dt_series.diff()
    seconds_diff = time_deltas.dt.total_seconds().fillna(0)
    return [int(s) for s in seconds_diff]

def categorize_time_diff(seconds):
    if seconds < 12: return 'aaaimmediatebbb'
    elif seconds < 32: return 'aaaquickbbb'
    elif seconds < 123: return 'aaaslowbbb'
    else: return 'aaadawdlebbb'

dict_shipper = {
    1: 'aaabuyerbbb',
    2: 'aaasellerbbb'
}

def preprocess_dataset(seq_dataset, args):
    if args.preprocess_method == 'takemoto':
        if args.all_categories:
            # replace sharp sign with a space
            seq_dataset['category_name'] = seq_dataset['category_name'].map(
                lambda cat_list: [x.replace('#', ' ') for x in cat_list])
        else:
            seq_dataset['category_name'] = seq_dataset['category_name'].map(
                lambda cat_list: [x.split('#')[0] or x.split('#')[1] or x.split('#')[2] for x in cat_list])
        def convert_category_id(cat_list):
            cat_list = [x.split('#')[0] or x.split('#')[1] or x.split('#')[2] for x in cat_list]
            return [int(float(x)) for x in cat_list]
        seq_dataset['category_id'] = seq_dataset['category_id'].map(convert_category_id)
        # convert ids to tokens and add to tokenizer
        if args.add_event_id:
            seq_dataset["event_id"] = seq_dataset["event_id"].map(
                lambda eve_list: [EVENT_ID_TO_TOKEN[x] for x in eve_list])

        if args.add_shipper_id:
            seq_dataset["shipper_id"] = seq_dataset["shipper_id"].map(
                lambda ship_list: [SHIPPER_ID_TO_TOKEN[x] for x in ship_list])
        # add special tokens to category_name
        append_tokens_to_cat(args, seq_dataset)
    elif args.preprocess_method == 'kawabata':
        # カテゴリ階層
        seq_dataset['category_name'] = seq_dataset['category_name'].apply(all_hierarchical_category)
        seq_dataset["category_id"] = seq_dataset["category_id"].apply(min_hierarchical_category)
        # イベントID
        seq_dataset["event_id"] = seq_dataset["event_id"].apply(lambda lst: [dict_event[x] for x in lst])
        # タイムスタンプ
        seq_dataset['time_diff_seconds'] = seq_dataset['stime'].apply(calculate_time_diffs)
        seq_dataset['time_label'] = seq_dataset['time_diff_seconds'].apply(
            lambda seq: [categorize_time_diff(s) for s in seq])
        # Shipper
        seq_dataset["shipper_id"] = seq_dataset["shipper_id"].apply(lambda lst: [dict_shipper[x] for x in lst])

        # イベントID * タイムスタンプ
        seq_dataset['event_time'] = seq_dataset.apply(
            lambda row: [a + b for a, b in zip(row['event_id'], row['time_label'])], axis=1)
        seq_dataset["event_time"] = seq_dataset["event_time"].apply(lambda lst: [s.replace("bbbaaa", "") for s in lst])
        # イベントID * Shipper
        seq_dataset['event_shipper'] = seq_dataset.apply(
            lambda row: [a + b for a, b in zip(row['event_id'], row['shipper_id'])], axis=1)
        seq_dataset["event_shipper"] = seq_dataset["event_shipper"].apply(
            lambda lst: [s.replace("bbbaaa", "") for s in lst])
        # タイムスタンプ * Shipper
        seq_dataset["time_shipper"] = seq_dataset.apply(
            lambda row: [a + b for a, b in zip(row['time_label'], row['shipper_id'])], axis=1)
        seq_dataset["time_shipper"] = seq_dataset["time_shipper"].apply(
            lambda lst: [s.replace("bbbaaa", "") for s in lst])
        # イベントID * タイムスタンプ * Shipper
        seq_dataset["event_time_shipper"] = seq_dataset.apply(
            lambda row: [a + b for a, b in zip(row['event_time'], row['shipper_id'])], axis=1)
        seq_dataset["event_time_shipper"] = seq_dataset["event_time_shipper"].apply(
            lambda lst: [s.replace("bbbaaa", "") for s in lst])

        # 追加属性をcategory_nameに結合
        target_cols = ['category_name', 'event_id', 'time_label', 'shipper_id',
                       'event_time', 'event_shipper', 'time_shipper', 'event_time_shipper']
        seq_dataset['category_name'] = seq_dataset[target_cols].apply(
            lambda row: [''.join(items) for items in zip(*row)], axis=1)
        # 以降の処理に用いる最終的なseq_dataset
        seq_dataset = seq_dataset[
            ['seq_user_id', 'name', 'category_name', 'brand_name', 'category_id', 'brand_id', 'item_id']]
    elif args.preprocess_method == "ogawa":
        raise NotImplementedError("Ogawa preprocessing method is not implemented yet.")

    return seq_dataset

def append_tokens_to_cat(args, seq_dataset):
    # add event_id and shipper_id to category_name
    if args.add_event_id or args.add_shipper_id:
        seq_dataset.reset_index(inplace=True)
        for index, row in enumerate(seq_dataset.itertuples(index=False, name="Row")):
            new_category_name = row.category_name.copy()
            if args.add_event_id:
                new_category_name = [cat + eve for cat, eve in zip(new_category_name, row.event_id)]
            if args.add_shipper_id:
                new_category_name = [cat + ship for cat, ship in zip(new_category_name, row.shipper_id)]
            if args.add_event_id and args.add_shipper_id:
                new_category_name = [cat + combine_tokens([eve, ship]) for cat, eve, ship in zip(
                    new_category_name, row.event_id, row.shipper_id)]
            seq_dataset.at[index, 'category_name'] = new_category_name


def add_special_tokens(args, seq_dataset, tokenizer):
    num_added = 0
    # convert ids to tokens and add to tokenizer
    if args.add_event_id:
        seq_dataset["event_id"] = seq_dataset["event_id"].map(
            lambda eve_list: [EVENT_ID_TO_TOKEN[x] for x in eve_list])
        num_added += tokenizer.add_special_tokens(list(EVENT_ID_TO_TOKEN.values()))
    if args.add_shipper_id:
        seq_dataset["shipper_id"] = seq_dataset["shipper_id"].map(
            lambda ship_list: [SHIPPER_ID_TO_TOKEN[x] for x in ship_list])
        num_added += tokenizer.add_special_tokens(list(SHIPPER_ID_TO_TOKEN.values()))
    if args.add_event_id and args.add_shipper_id:
        combined_tokens = []
        for event_id in EVENT_ID_TO_TOKEN.values():
            for shipper_id in SHIPPER_ID_TO_TOKEN.values():
                combined_tokens.append(combine_tokens([event_id, shipper_id]))
        num_added += tokenizer.add_special_tokens(combined_tokens)
    return num_added


def main(args):
    logging.info("Starting recommendation pipeline...")
    data_path = args.data_path
    data = pd.read_pickle(data_path)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    seq_dataset = pd.DataFrame.from_dict(data, orient='index')
    # seq_dataset = seq_dataset[
    #     ['seq_user_id', 'name', 'category_name', 'brand_name', 'category_id', 'brand_id', 'item_id', 'event_id']]
    logging.info("Before preprocessing:")
    logging.info(seq_dataset.iloc[0])
    seq_dataset = preprocess_dataset(seq_dataset, args)
    logging.info("After preprocessing:")
    logging.info(seq_dataset.iloc[0])
    logging.info(seq_dataset.iloc[0]["category_name"])
    _, test_df = train_test_split(
        seq_dataset, test_size=args.test_frac, random_state=args.seed)
    val_df, test_df = train_test_split(
        test_df, test_size=0.5, random_state=args.seed)

    test_dataset = UserItemInteractionDataset(interactions=test_df)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_batch_item_val, tokenizer=tokenizer, return_user_id=True),
        drop_last=True,
    )

    model = ThreeTower(
        Encoder(  # user_encoder
            EncoderLayer(
                config.D_MODEL,
                MultiHeadedAttention(config.NUM_HEADS, config.D_MODEL),
                PositionwiseFeedForward(
                    config.D_MODEL, config.D_FF, config.DROPOUT),
                config.DROPOUT,
            ),
            config.NUM_STACKS,
        ),
        nn.Sequential(  # user_encoder_embed
            UserEmbeddings(
                vocab_size=config.BPE_VOCAB_LIMIT + args.num_added_tokens,
                d_model=config.D_MODEL,
                padding_idx=tokenizer.token_to_id(MASK_TOKEN),
                max_norm=config.MAX_NORM,
            ),
            nn.Unflatten(0, (config.BATCH_SIZE, -1)),
            PositionalEncoding(config.D_MODEL, config.DROPOUT,
                               config.POSITION_MAX_LEN),
        ),
        Decoder(  # user_decoder
            DecoderLayer(
                config.D_MODEL,
                MultiHeadedAttention(config.NUM_HEADS, config.D_MODEL),
                MultiHeadedAttention(config.NUM_HEADS, config.D_MODEL),
                PositionwiseFeedForward(
                    config.D_MODEL, config.D_FF, config.DROPOUT),
                config.DROPOUT,
            ),
            config.NUM_STACKS,
        ),
        nn.Sequential(  # user_decoder_embed
            ItemEmbeddings(
                vocab_size=config.BPE_VOCAB_LIMIT + args.num_added_tokens,
                d_model=config.D_MODEL,
                padding_idx=tokenizer.token_to_id(MASK_TOKEN),
                max_norm=config.MAX_NORM,
            ),
            nn.Unflatten(0, (config.BATCH_SIZE, -1)),
            PositionalEncoding(config.D_MODEL, config.DROPOUT,
                               config.POSITION_MAX_LEN),
        ),
        Encoder(  # item_encoder
            EncoderLayer(
                config.D_MODEL,
                MultiHeadedAttention(config.NUM_HEADS, config.D_MODEL),
                PositionwiseFeedForward(
                    config.D_MODEL, config.D_FF, config.DROPOUT),
                config.DROPOUT,
            ),
            config.NUM_STACKS,
        ),
        nn.Sequential(  # item_encoder_embed
            ItemEmbeddings(
                vocab_size=config.BPE_VOCAB_LIMIT + args.num_added_tokens,
                d_model=config.D_MODEL,
                padding_idx=tokenizer.token_to_id(MASK_TOKEN),
                max_norm=config.MAX_NORM,
            ),
            nn.Unflatten(0, (config.BATCH_SIZE, -1)),
        ),
    )

    # load model from args.model_path
    model_path = pathlib.Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    logging.info(f"The device is: {config.DEVICE}")

    with torch.no_grad():
        model.eval()
        critic = Evaluator(
            batch_size=config.BATCH_SIZE,
            num_eval_seq=config.NUM_EVAL_SEQ,
            model=model,
            d_model=config.D_MODEL,
            lookup_size=config.LOOKUP_SIZE,
            val_loader=test_loader,
            eval_ks=config.EVAL_Ks,
            tokenizer=tokenizer,
            out_dir=args.metrics_path,
        )
        critic.recommend(desc="trained")
    logging.info("Recommendation completed.")


def parse_args(description):

    parser = ArgumentParser(description=description)
    parser.add_argument('--model_path', type=str, default='./checkpoint/')
    parser.add_argument('--metrics_path', type=str, default='./metrics/')
    parser.add_argument('--tokenizer_path', type=str,
                        default='tokenizer.json')
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--max_len", type=int, default=22)
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--test_frac", type=int, default=0.3)
    parser.add_argument("--add_event_id", action='store_true',
                        help="Add event_id to category_name")
    parser.add_argument("--add_shipper_id", action='store_true',
                        help="Add shipper_id to category_name")
    parser.add_argument("--all_categories", action='store_true',)
    parser.add_argument("--num_added_tokens", type=int, default=0,)
    parser.add_argument('--preprocess_method', choices=['kawabata', 'takemoto', 'ogawa'],
                        required=True,),
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='\n%(message)s')
    sys.exit(main(parse_args("Run training pipeline.")))
