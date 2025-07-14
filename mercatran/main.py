import logging
import os
import pathlib
import sys
from argparse import ArgumentParser
from functools import partial

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
    "item_view": "<item_view>",
    "item_like":  "<item_like>",
    "item_add_to_cart_tap":  "<item_add_to_cart_tap>",
    "offer_make":  "<offer_make>",
    "buy_start":  "<buy_start>",
    "buy_comp":  "<buy_comp>",
}

SHIPPER_ID_TO_TOKEN = {
    1: "<Buyer>",
    2: "<Seller>"
}

def preprocess_before_training_tokenizer(seq_dataset, args):
    if args.all_categories:
        # replace sharp sign with a space
        seq_dataset['category_name'] = seq_dataset['category_name'].map(
            lambda cat_list: [x.replace('#', ' ', regex=False) for x in cat_list])
    else:
        seq_dataset['category_name'] = seq_dataset['category_name'].map(
            lambda cat_list: [x.split('#')[0] or x.split('#')[1] or x.split('#')[2] for x in cat_list])
    seq_dataset['category_id'] = seq_dataset['category_id'].map(
        lambda cat_list: [float(x.split('#')[0]) or float(x.split('#')[1]) or float(x.split('#')[2]) for x in cat_list])

def combine_tokens(tokens, trim=True):
    if isinstance(tokens, pd.Series):
        tokens = tokens.tolist()
    if trim:
        # Remove the first and last characters (e.g., '<' and '>')
        tokens = [token[1:-1] for token in tokens]
    combined = ','.join(tokens)
    return "<" + combined + ">"

def preprocess_after_training_tokenizer(seq_dataset, tokenizer, args):
    # add special tokens to category_name
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

    # add event_id and shipper_id to category_name
    if args.add_event_id or args.add_shipper_id:
        seq_dataset.reset_index(inplace=True)
        for index, row in enumerate(seq_dataset.itertuples(index=False, name="Row")):
            new_category_name = row.category_name.copy()
            if args.add_event_id:
                new_category_name = [cat + eve for cat,eve in zip(new_category_name, row.event_id)]
            if args.add_shipper_id:
                new_category_name = [cat + ship for cat, ship in zip(new_category_name, row.shipper_id)]
            if args.add_event_id and args.add_shipper_id:
                new_category_name = [cat + combine_tokens([eve, ship]) for cat, eve, ship in zip(
                    new_category_name, row.event_id, row.shipper_id)]
            seq_dataset.at[index, 'category_name'] = new_category_name

    return num_added


def main(args):
    model_dir = pathlib.Path(args.save_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_path
    data = pd.read_pickle(data_path)
    seq_dataset = pd.DataFrame.from_dict(data, orient='index')
    # seq_dataset = seq_dataset[
    #     ['seq_user_id', 'name', 'category_name', 'brand_name', 'category_id', 'brand_id', 'item_id', 'event_id']]
    preprocess_before_training_tokenizer(seq_dataset, args)
    logging.info(seq_dataset.iloc[0])
    tokenizer = train_tokenizer(df_train=seq_dataset)
    num_added_tokens = preprocess_after_training_tokenizer(seq_dataset, tokenizer, args)
    tokenizer.save(os.path.join(args.save_path, args.tokenizer_save_name))
    logging.info(seq_dataset.iloc[0])
    logging.info(seq_dataset.iloc[0]["category_name"])
    logging.info(f"Number of added tokens: {num_added_tokens}, New tokenizer size: {tokenizer.get_vocab_size()}")
    train_df, test_df = train_test_split(
        seq_dataset, test_size=args.test_frac, random_state=args.seed)
    val_df, test_df = train_test_split(
        test_df, test_size=0.5, random_state=args.seed)

    train_dataset = UserItemInteractionDataset(interactions=train_df)
    val_dataset = UserItemInteractionDataset(interactions=val_df)
    test_dataset = UserItemInteractionDataset(interactions=test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_batch_item, tokenizer=tokenizer),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_batch_item_val, tokenizer=tokenizer),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_batch_item_val, tokenizer=tokenizer),
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
                vocab_size=config.BPE_VOCAB_LIMIT + num_added_tokens,
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
                vocab_size=config.BPE_VOCAB_LIMIT + num_added_tokens,
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
                vocab_size=config.BPE_VOCAB_LIMIT + num_added_tokens,
                d_model=config.D_MODEL,
                padding_idx=tokenizer.token_to_id(MASK_TOKEN),
                max_norm=config.MAX_NORM,
            ),
            nn.Unflatten(0, (config.BATCH_SIZE, -1)),
        ),
    )

    model = model_initialization(model)
    model.to(config.DEVICE)
    logging.info(f"The device is: {config.DEVICE}")
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.BASE_LR,
        betas=config.BETAS,
        eps=config.EPS,
        weight_decay=config.WEIGHT_DECAY,
        amsgrad=False,
        maximize=False,
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, config.D_MODEL, factor=1, warmup=config.WARMUP_STEPS
        ),
    )

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = []
        for i, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch[{epoch}] Training batch")
        ):
            user, user_mask, item, item_mask, item_y, item_mask_y = batch
            user_embed, item_embed = model(
                user,
                user_mask.unsqueeze(-2),
                item,
                create_user_target_mask(item_mask.unsqueeze(-2)),
                item_y,
                create_item_encoder_mask(item_mask_y.unsqueeze(-2)),
            )
            target = torch.arange(user_embed.size(
                0) * user_embed.size(1)).to(config.DEVICE)
            user_embed = user_embed.view(-1, config.D_MODEL)
            item_embed = item_embed.view(-1, config.D_MODEL)
            res = torch.matmul(user_embed, item_embed.t())
            loss = ce_loss(res, target)
            loss.backward()
            if i % config.ACCUM_ITER == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            total_loss.append(loss.item())

            del loss

        train_loss = sum(total_loss) / len(total_loss)
        logging.info(f"Epoch: {epoch}")
        logging.info(f"Train Loss: {train_loss}")
        if epoch % config.EVAL_EPOCHS == 0:
            torch.save(model.state_dict(), os.path.join(
                args.save_path, f"{epoch}-model.pt"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                model.eval()
                critic = Evaluator(
                    batch_size=config.BATCH_SIZE,
                    num_eval_seq=config.NUM_EVAL_SEQ,
                    model=model,
                    d_model=config.D_MODEL,
                    lookup_size=config.LOOKUP_SIZE,
                    val_loader=val_loader,
                    eval_ks=config.EVAL_Ks,
                    tokenizer=tokenizer,
                    out_dir=args.metrics_path,
                )
                _ = critic.evaluate(epoch_train=epoch, desc="val")
                del critic

    logging.info("Final evaluation with test set")
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
        _ = critic.evaluate(epoch_train=epoch, desc="final_test")


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
    parser.add_argument("--add_event_id", action='store_true',
                        help="Add event_id to category_name")
    parser.add_argument("--add_shipper_id", action='store_true',
                        help="Add shipper_id to category_name")
    parser.add_argument("--all_categories", action='store_true',)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='\n%(message)s')
    sys.exit(main(parse_args("Run training pipeline.")))
