import sys
import logging
import os.path as osp
import os
import pickle
from argparse import ArgumentParser
import random

from mercatran.data import sequence_dataset_b


def parse_args(description):

    parser = ArgumentParser(description=description)
    parser.add_argument('--save_path', type=str, default='./checkpoint/')
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--max_len", type=int, default=22)
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--test_frac", type=int, default=0.3)
    parser.add_argument("--num_df", type=int, default=1)
    args = parser.parse_args()
    return args

def main(args):
    data_path = args.data_path
    seq_dataset = sequence_dataset_b(path=data_path, num_df=args.num_df)
    file_name = osp.join(args.save_path, "b_data.pkl")
    random.seed(args.seed)
    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(file_name, "wb") as f:
        pickle.dump(seq_dataset, f)
    logging.info(f"Saved B data to {file_name}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='\n%(message)s')
    sys.exit(main(parse_args("Save B data for New project")))