import logging
import os
import pathlib
import sys
from argparse import ArgumentParser
import os.path as osp
from tqdm.auto import tqdm
import pandas as pd
import csv


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

def preprocess_dataset(seq_dataset, args):
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


def retrieve_item_info(item_id, user_id, data_dict):
    row = data_dict[user_id]
    index = row['item_id'].index(item_id)
    return {
        'item_id': item_id,
        'name': row['name'][index],
        'category_name': row['category_name'][index],
        'brand_name': row['brand_name'][index],
        'price': row['price'][index],
        'item_condition_id': row['item_condition_id'][index],
        'size_id': row['size_id'][index],
        'shipper_id': row['shipper_id'][index],
    }

def save_recommend_items(data_dict, rec_row, base_dir):
    user_id = rec_row.user_id
    seq_user_dir = osp.join(base_dir, str(user_id))
    os.makedirs(seq_user_dir, exist_ok=True)

    with open(osp.join(seq_user_dir, "recommend_items.csv"), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'item_id', 'name', 'category_name', 'brand_name',
            'price', 'item_condition_id', 'size_id', 'shipper_id'
        ])
        writer.writeheader()
        for n in range(1, 6):
            item_id = getattr(rec_row, f'pred_item_id_{n}')
            user_item_id = getattr(rec_row, f'pred_user_id_{n}')
            item_info = retrieve_item_info(item_id, user_item_id, data_dict)
            writer.writerow(item_info)

def save_sequence_items(user_id, data_dict, base_dir):
    seq_user_dir = osp.join(base_dir, str(user_id))
    os.makedirs(seq_user_dir, exist_ok=True)
    seq_dict = data_dict[user_id].copy()
    seq_dict.pop('seq_user_id', None)  # Remove user_id from the sequence dict
    seq_dict.pop('sequence_length', None)  # Remove sequence length if exists

    seq_df = pd.DataFrame(seq_dict)
    seq_df.to_csv(osp.join(seq_user_dir, "sequence_items.csv"), index=False)





def main(args):
    logging.info("Starting recommendation pipeline...")
    os.makedirs(args.output_dir, exist_ok=True)
    data_path = args.data_path
    data_dict = pd.read_pickle(data_path)
    recommend_df = pd.read_csv(args.recommend_csv_path)
    logging.info(recommend_df.head())

    for row in tqdm(recommend_df.itertuples(index=False, name="Row"), desc="Processing recommendations"):
        save_recommend_items(data_dict, row, args.output_dir)
        save_sequence_items(row.user_id, data_dict, args.output_dir)

    logging.info("Recommendation items saved successfully.")





def parse_args(description):
    parser = ArgumentParser(description=description)
    parser.add_argument('--recommend_csv_path', type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="recommend_items")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--max_len", type=int, default=22)
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--test_frac", type=int, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='\n%(message)s')
    sys.exit(main(parse_args("Retrieving item information.")))
