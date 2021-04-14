import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset_intent import SeqClsDataset
from utils_intent import Vocab

import tensorflow as tf
import numpy as np
# from pytorch_model import SeqClassifier
from tf_model_intent import SeqClassifier
from torch.utils.data import DataLoader
import csv


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    data_loader = DataLoader(
        dataset,
        batch_size = 15000,
        shuffle = False,
        collate_fn = dataset.collate_fn
    )
    for d in data_loader:
        test_X = d['encoded_split_text']

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SeqClassifier(
        mode='intent',               # Added by me
        text_len=args.max_len,       # Added by me
        embeddings=embeddings,
        batch_size=args.batch_size,  # Added by me
        dropout=args.dropout,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        num_class=dataset.num_classes,
    )
    model.build((128, 28))

    # load weights into model
    model.load_weights(args.ckpt_path)

    # TODO: predict dataset
    pred_Y_one_hot = model.predict(
        x=test_X,
        batch_size=args.batch_size,
        verbose=1,
    )
    pred_Y_label = []

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'intent'])
        for i, one_hot in enumerate(pred_Y_one_hot):
            pred_Y_label.append(dataset.idx2label(list(one_hot).index(max(one_hot))))
            writer.writerow([f"test-{i}", pred_Y_label[i]])


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=28)

    # model
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
