import copy
import logging
import os
import random
from collections import namedtuple
from typing import Callable, Iterator, List, Union

import config
import numpy as np
import pandas as pd
import torch
from text import (
    DEFAULT_TOKEN,
    END_TOKEN,
    MASK_TOKEN,
    START_TOKEN,
    preprocess_text,
)
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def bpe_text_pipeline(
    input: Union[List[str], str],
    tokenizer: Callable[[str], List[str]],
) -> List[int]:
    """A utility to convert a list of strings or a string into tokens 
    defined by the trained tokenizer"""
    return (
        tokenizer.encode(preprocess_text(input)).ids
        if input.strip()
        else [tokenizer.token_to_id(DEFAULT_TOKEN)]
    )


def batch_iterator(
        df: pd.DataFrame,
        include_brand_cat=True
) -> Iterator[List[str]]:
    for _, row in df.iterrows():
        title = row["name"][0] if "name" in row and row["name"][0] else ""
        brand_name = (
            row["brand_name"][0] if "brand_name" in row and row["brand_name"][0] else ""  # noqa: E501
        )
        cat_name = (
            row["category_name"][0]
            if "category_name" in row and row["category_name"][0]
            else ""
        )
        yield preprocess_text(title) + " " + preprocess_text(
            brand_name
        ) + " " + preprocess_text(
            cat_name) if include_brand_cat else preprocess_text(
            title
        )


def train_tokenizer(df_train: pd.DataFrame) -> Tokenizer:
    tokenizer = Tokenizer(model=models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=config.BPE_VOCAB_LIMIT,
        special_tokens=[START_TOKEN, END_TOKEN, MASK_TOKEN, DEFAULT_TOKEN],
    )
    tokenizer.train_from_iterator(batch_iterator(df=df_train), trainer=trainer)
    logger.info(f"Constructed vocab of size: {config.BPE_VOCAB_LIMIT}")
    return tokenizer


def sequence_dataset(path, min_seq_len=10):
    data_files = os.listdir(path)
    df = pd.concat(
        [
            pd.read_parquet(
                os.path.join(path, file)
            ) for file in data_files
        ],
        ignore_index=True
    )
    df['seq_user_id'] = df['user_id'].astype(
        str) + "_" + df['sequence_id'].astype(str)
    df["category_name"] = df[config.CATEGORY_NAME_HIERARCHY].bfill(
        axis=1).iloc[:, 0]
    df["category_id"] = df[config.CATEGORY_ID_HIERARCHY].bfill(
        axis=1).iloc[:, 0]
    df = df.drop(config.CATEGORY_NAME_HIERARCHY +
                 config.CATEGORY_ID_HIERARCHY, axis=1)

    sequences = df.groupby('seq_user_id', as_index=False).agg(
        {
            'name': list,
            'category_name': list,
            'brand_name': list,
            'category_id': list,
            'brand_id': list,
            'item_id': list
        }
    )
    return sequences[sequences['name'].apply(len) >= min_seq_len]


def sequence_dataset_b(path, min_seq_len=10, sample_prob=0.11, num_df=1,
                       concat_category=False, include_event_id=False, sort_seq=False,
                       user_seq_ids_prev=None):
    event_id_table = {
        "item_view": 0,
        "item_like": 1,
        "item_add_to_cart_tap": 2,
        "offer_make": 3,
        "buy_start": 4,
        "buy_comp": 5,
    }
    print("creating df")
    data_files = os.listdir(path)
    chunk_size = len(data_files) // num_df
    data_files_chunked = [data_files[i:i + chunk_size] for i in range(0, len(data_files), chunk_size)]
    result_dict = dict()
    rejected_ids = set()
    for chunk in data_files_chunked:
        print("processing a chunk")
        df = pd.concat(
            [
                pd.read_parquet(
                    os.path.join(path, file)
                ) for file in chunk
            ],
            ignore_index=True
        )
        df['seq_user_id'] = df['user_id'].astype(
            str) + "_" + df['sequence_id'].astype(str)
        if concat_category:
            df["category_name"] = df[config.CATEGORY_NAME_HIERARCHY].astype('string').fillna("").agg('#'.join, axis=1)
            df["category_id"] = df[config.CATEGORY_ID_HIERARCHY].astype('string').fillna("").agg('#'.join, axis=1)
        else:
            df["category_name"] = df[config.CATEGORY_NAME_HIERARCHY].bfill(
                axis=1).iloc[:, 0]
            df["category_id"] = df[config.CATEGORY_ID_HIERARCHY].bfill(
                axis=1).iloc[:, 0]
        df = df.drop(config.CATEGORY_NAME_HIERARCHY +
                     config.CATEGORY_ID_HIERARCHY, axis=1)
        if include_event_id:
            # to raise an error if event_type is not in the df, we pass a function instead of the dict
            df["event_id"] = df["event_id"].map(lambda x: event_id_table[x])
        # convert TimeStamp object into string to reduce size
        df["stime"] = df["stime"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        agg_func_dict = {
                'name': list,
                'category_name': list,
                'brand_name': list,
                'category_id': list,
                'brand_id': list,
                'item_id': list,
                'event_id': list,
                'price': list,
                'item_condition_id': list,
                'size_id': list,
                'shipper_id': list,
                'stime': list,
                'sequence_length': 'first'
        }
        if include_event_id:
            agg_func_dict['event_id'] = list
        sequences = df.groupby('seq_user_id', as_index=False).agg(
            agg_func_dict
        )
        del df
        filter_seq = {}
        print("initial dict for chunk is created")
        for row in tqdm(sequences.itertuples(index=False, name='Row'), desc="Filtering sequences"):
            if getattr(row, "seq_user_id") in result_dict:
                prev_record = result_dict[getattr(row, "seq_user_id")]
                new_record = {key: value + getattr(row, key) for key,value in prev_record.items() if key != 'sequence_length'}
                new_record['sequence_length'] = prev_record['sequence_length']
                filter_seq[prev_record["seq_user_id"]] = new_record
                continue
            newly_accepted = False
            if user_seq_ids_prev is not None:
                if getattr(row, "seq_user_id") in user_seq_ids_prev:
                    newly_accepted = True
            elif (getattr(row, "seq_user_id") not in rejected_ids) and random.random() <= sample_prob:  # keep roughly 10% of data
                newly_accepted = True
            if newly_accepted:
                last_record = row._asdict()
                filter_seq[last_record["seq_user_id"]] = last_record
            else:
                rejected_ids.add(getattr(row, "seq_user_id"))
        result_dict.update(filter_seq)
    print("lastly, filtering result_dict with min_seq_len")
    for key in list(result_dict.keys()):
        if len(result_dict[key]["name"]) < min_seq_len:
            del result_dict[key]
            continue
        if sort_seq:
            # sort by stime
            sorted_indices = np.argsort(result_dict[key]["stime"])
            for k in result_dict[key].keys():
                if k in ["sequence_length", "seq_user_id"]:
                    continue
                if len(result_dict[key][k]) != len(sorted_indices):
                    print(result_dict[key])
                    raise ValueError(
                        f"Length mismatch for key {k} in sequence {key}. "
                        f"Expected {len(sorted_indices)}, got {len(result_dict[key][k])}"
                    )
                result_dict[key][k] = [result_dict[key][k][i] for i in sorted_indices]
    return result_dict


def create_start_token_sequence(tokenizer, batch_size):
    start_dict = {"tokens": [], "offsets": [0]}
    for _ in range(batch_size):
        add_token(start_dict, tokenizer=tokenizer, token_type=START_TOKEN)
    start_dict["offsets"] = (
        torch.tensor(start_dict["offsets"][:-1]
                     ).cumsum(dim=0).to(config.DEVICE)
    )
    return (
        torch.cat(start_dict["tokens"]).to(
            config.DEVICE), start_dict["offsets"]
    )


def add_token(collec, tokenizer: Tokenizer, token_type=START_TOKEN):
    collec["tokens"].append(torch.tensor([tokenizer.token_to_id(token_type)]))
    collec["offsets"].append(1)  # size of the token


def collate_batch_item(batch: torch.Tensor, tokenizer: Tokenizer):
    item_dict = {"tokens": [], "offsets": [0]}
    item_dict_y = {"tokens": [], "offsets": [0]}
    user_dict = {"tokens": [], "offsets": [0]}
    user_mask, item_mask = [], []

    for category, brand, title, _, _, _, _ in batch:
        # add start token to start of each sequence, see below for user
        add_token(item_dict, tokenizer=tokenizer, token_type=START_TOKEN)
        for ti, br, ca in zip(
            title[-config.NUM_EVAL_SEQ:],
            brand[-config.NUM_EVAL_SEQ:],
            category[-config.NUM_EVAL_SEQ:],
        ):
            concat_feat = preprocess_text(ti + " " + br + " " + ca)
            processed_feat = bpe_text_pipeline(concat_feat, tokenizer)
            assert processed_feat, "The text pipeline failed for features"
            if not processed_feat:
                processed_feat = [tokenizer.token_to_id(DEFAULT_TOKEN)]
            tensor_feat = torch.tensor(processed_feat, dtype=torch.long)
            item_dict["tokens"].append(tensor_feat)
            item_dict["offsets"].append(tensor_feat.size(0))
        # add end token to end of the item target sequence
        # add_token(item_dict, token_type=END_TOKEN)
        for ti, br, ca in zip(
            title[-config.NUM_EVAL_SEQ:],
            brand[-config.NUM_EVAL_SEQ:],
            category[-config.NUM_EVAL_SEQ:],
        ):
            concat_feat = preprocess_text(ti + " " + br + " " + ca)
            processed_feat = bpe_text_pipeline(concat_feat, tokenizer)
            assert processed_feat, "The text pipeline failed for features"
            if not processed_feat:
                processed_feat = [tokenizer.token_to_id(DEFAULT_TOKEN)]
            tensor_feat = torch.tensor(processed_feat, dtype=torch.long)
            item_dict_y["tokens"].append(tensor_feat)
            item_dict_y["offsets"].append(tensor_feat.size(0))
        # add end token to end of the item target sequence
        add_token(item_dict_y, tokenizer=tokenizer, token_type=END_TOKEN)

        assert len(category) == len(brand) == len(
            title), "Batching not working"
        # add mask tokens to item sequence if needed
        item_mask.append(
            [True for _ in range(config.NUM_EVAL_SEQ + 1)]
        )  # 1 -> start or end token
        add_token(user_dict, tokenizer=tokenizer, token_type=START_TOKEN)
        for ti, br, ca in zip(
            title[: -config.NUM_EVAL_SEQ],
            brand[: -config.NUM_EVAL_SEQ],
            category[: -config.NUM_EVAL_SEQ],
        ):
            concat_feat = preprocess_text(ti + " " + br + " " + ca)
            processed_feat = bpe_text_pipeline(concat_feat, tokenizer)
            assert processed_feat, "The text pipeline failed for features"
            if not processed_feat:
                processed_feat = [tokenizer.token_to_id(DEFAULT_TOKEN)]
            tensor_feat = torch.tensor(processed_feat, dtype=torch.long)
            user_dict["tokens"].append(tensor_feat)
            user_dict["offsets"].append(tensor_feat.size(0))
        add_token(user_dict, tokenizer=tokenizer, token_type=END_TOKEN)

        user_mask.append(
            [True for _ in range(len(category) - config.NUM_EVAL_SEQ + 2)]
        )  # 2 -> start + end tokens

        if len(category) - config.NUM_EVAL_SEQ < config.MODEL_SEQ_LEN:
            for _ in range(
                config.MODEL_SEQ_LEN - (len(category) - config.NUM_EVAL_SEQ)
            ):
                add_token(user_dict, tokenizer=tokenizer,
                          token_type=MASK_TOKEN)
                user_mask[-1].append(False)

    item_dict["offsets"] = (
        torch.tensor(item_dict["offsets"][:-1]).cumsum(dim=0).to(config.DEVICE)
    )
    item_dict_y["offsets"] = (
        torch.tensor(item_dict_y["offsets"][:-1]
                     ).cumsum(dim=0).to(config.DEVICE)
    )
    user_dict["offsets"] = (
        torch.tensor(user_dict["offsets"][:-1]).cumsum(dim=0).to(config.DEVICE)
    )
    item_mask_y = copy.deepcopy(item_mask)

    return (
        (torch.cat(user_dict["tokens"]).to(
            config.DEVICE), user_dict["offsets"]),
        torch.from_numpy(np.array(user_mask)).to(config.DEVICE),
        (torch.cat(item_dict["tokens"]).to(
            config.DEVICE), item_dict["offsets"]),
        torch.from_numpy(np.array(item_mask)).to(config.DEVICE),
        (torch.cat(item_dict_y["tokens"]).to(
            config.DEVICE), item_dict_y["offsets"]),
        torch.from_numpy(np.array(item_mask_y)).to(config.DEVICE),
    )


def collate_batch_item_val(batch: torch.Tensor, tokenizer: Tokenizer):
    user_dict = {"tokens": [], "offsets": [0]}
    item_dict_y = {"tokens": [], "offsets": [0]}
    category_id_dict = {"tokens": []}
    brand_id_dict = {"tokens": []}
    item_id_dict = {"tokens": []}
    user_mask, item_mask = [], []

    for category, brand, title, category_id, brand_id, item_id, _ in batch:
        # add start token to the start of user sequence
        add_token(user_dict, tokenizer=tokenizer, token_type=START_TOKEN)
        # use only the first N - config.NUM_EVAL_SEQ
        for ti, br, ca in zip(
            title[: -config.NUM_EVAL_SEQ],
            brand[: -config.NUM_EVAL_SEQ],
            category[: -config.NUM_EVAL_SEQ],
        ):
            concat_feat = preprocess_text(ti + " " + br + " " + ca)
            processed_feat = bpe_text_pipeline(concat_feat, tokenizer)
            assert processed_feat, "The text pipeline failed for features"
            if not processed_feat:
                processed_feat = [tokenizer.token_to_id(DEFAULT_TOKEN)]
            tensor_feat = torch.tensor(processed_feat, dtype=torch.long)
            user_dict["tokens"].append(tensor_feat)
            user_dict["offsets"].append(tensor_feat.size(0))
        add_token(user_dict, tokenizer=tokenizer, token_type=END_TOKEN)

        assert len(category) == len(brand) == len(
            title), "Batching not working"
        user_mask.append(
            [True for _ in range(len(category) - config.NUM_EVAL_SEQ + 2)]
        )  # 2 -> start + end tokens

        if len(title) - config.NUM_EVAL_SEQ < config.MODEL_SEQ_LEN:
            for _ in range(config.MODEL_SEQ_LEN - (len(title) - config.NUM_EVAL_SEQ)):  # noqa: E501
                add_token(user_dict, tokenizer=tokenizer,
                          token_type=MASK_TOKEN)
                user_mask[-1].append(False)

        for ti, br, ca in zip(
            title[-config.NUM_EVAL_SEQ:],
            brand[-config.NUM_EVAL_SEQ:],
            category[-config.NUM_EVAL_SEQ:],
        ):
            concat_feat = preprocess_text(ti + " " + br + " " + ca)
            processed_feat = bpe_text_pipeline(concat_feat, tokenizer)
            assert processed_feat, "The text pipeline failed for features"
            if not processed_feat:
                processed_feat = [tokenizer.token_to_id(DEFAULT_TOKEN)]
            tensor_feat = torch.tensor(processed_feat, dtype=torch.long)
            item_dict_y["tokens"].append(tensor_feat)
            item_dict_y["offsets"].append(tensor_feat.size(0))

        item_mask.append([True for _ in range(config.NUM_EVAL_SEQ)])

        category_id_dict["tokens"].append(
            torch.tensor([category_id[-config.NUM_EVAL_SEQ:]],
                         dtype=torch.long)
        )
        brand_id_dict["tokens"].append(
            torch.tensor([brand_id[-config.NUM_EVAL_SEQ:]], dtype=torch.long)
        )

        item_id_dict["tokens"].append(
            torch.tensor(
                [item_id[-config.NUM_EVAL_SEQ:]],
                dtype=torch.long,
            )
        )

    item_dict_y["offsets"] = (
        torch.tensor(item_dict_y["offsets"][:-1]
                     ).cumsum(dim=0).to(config.DEVICE)
    )
    user_dict["offsets"] = (
        torch.tensor(user_dict["offsets"][:-1]).cumsum(dim=0).to(config.DEVICE)
    )

    return (
        (torch.cat(user_dict["tokens"]).to(
            config.DEVICE), user_dict["offsets"]),
        torch.from_numpy(np.array(user_mask)).to(config.DEVICE),
        (torch.cat(item_dict_y["tokens"]).to(
            config.DEVICE), item_dict_y["offsets"]),
        torch.from_numpy(np.array(item_mask)).to(config.DEVICE),
        torch.cat(category_id_dict["tokens"], dim=0).to(config.DEVICE),
        torch.cat(brand_id_dict["tokens"], dim=0).to(config.DEVICE),
        torch.cat(item_id_dict["tokens"], dim=0).to(config.DEVICE),
    )


def separate_event_id(batch):
    new_batch = []
    event_ids = []
    for item in batch:
        new_batch.append(item[:-1])
        # pad the event_ids to MODEL_SEQ_LEN + 2 # (start and end tokens)
        event_ids += [config.EVENT_ID_PADDING_IDX] + item[-1][:-config.NUM_EVAL_SEQ]
        if len(item[-1]) - config.NUM_EVAL_SEQ < config.MODEL_SEQ_LEN:
            event_ids += [config.EVENT_ID_PADDING_IDX] * (config.MODEL_SEQ_LEN - len(item[-1]) + config.NUM_EVAL_SEQ)
        event_ids += [config.EVENT_ID_PADDING_IDX]
    return new_batch, event_ids


def collate_batch_item_wrapper(batch: torch.Tensor, tokenizer: Tokenizer):
    batch, event_ids = separate_event_id(batch)
    event_ids = torch.tensor(event_ids, dtype=torch.long).to(config.DEVICE)
    return collate_batch_item(batch, tokenizer) + (event_ids,)


def collate_batch_item_val_wrapper(batch: torch.Tensor, tokenizer: Tokenizer):
    batch, event_ids = separate_event_id(batch)
    event_ids = torch.tensor(event_ids, dtype=torch.long).to(config.DEVICE)
    return collate_batch_item_val(batch, tokenizer) + (event_ids,)


class UserItemInteractionDataset(Dataset):
    def __init__(self, interactions: pd.DataFrame, return_event_id=False):
        self.interactions = interactions
        self.return_event_id = return_event_id

    def __len__(self):
        return len(self.interactions)

    def _data_helper(self, events, tag):
        return [event if event else "" for event in events[tag]]

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = row['seq_user_id']
        category = self._data_helper(row, "category_name")
        brand = self._data_helper(row, "brand_name")
        title = self._data_helper(row, "name")
        category_id = [
            event if event else 0 for event in row["category_id"]
        ]
        brand_id = [event if event else 0 for event in row["brand_id"]]
        item_id = [event if event else 0 for event in row["item_id"]]
        if self.return_event_id:
            return category, brand, title, category_id, brand_id, item_id, user_id, row['event_id']
        else:
            return category, brand, title, category_id, brand_id, item_id, user_id