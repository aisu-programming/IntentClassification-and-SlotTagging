import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset_slot import SeqClsDataset
from utils_slot import Vocab

import tensorflow as tf
import numpy as np
from tf_model_slot import SeqClassifier
from train_slot_test import create_masks
from torch.utils.data import DataLoader
import csv
from tqdm import tqdm


def test_step(model, test_X, text_len):

    test_batches = []
    for i in range(len(test_X)//128+1):
        if i == len(test_X)//128:
            test_batches.append(test_X[i*128:])
        else:
            test_batches.append(test_X[i*128:(i+1)*128])

    pred_Y = tf.convert_to_tensor(np.empty((0, 37), dtype=np.float32))
    for x in tqdm(test_batches, total=len(test_batches)):
        output_id = tf.convert_to_tensor(np.concatenate([np.array([1], dtype=np.float32)[np.newaxis, :]]*x.shape[0]))
        # output = tf.convert_to_tensor(np.concatenate([np.array(
        #     [ [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ] ]
        # , dtype=np.float32)[np.newaxis, :]]*x.shape[0]))
        for i in range(text_len-1):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(x, output_id)

            predictions = model(x, output_id, False, enc_padding_mask, combined_mask, dec_padding_mask)
            # predictions = model(x, output_id, False, None, None, None)
            predictions = predictions[:, -1:, :]

            predicted_id = tf.argmax(predictions, axis=-1)
            predicted_id = tf.cast(predicted_id, dtype=tf.float32)

            # output    = tf.concat([output, predictions], axis=-2)
            output_id = tf.concat([output_id, predicted_id], axis=-1)

        pred_Y = tf.concat([pred_Y, output_id], axis=0)
    
    return pred_Y


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, tag2idx, args.max_len)
    data_loader = DataLoader(
        dataset,
        batch_size = 15000,
        shuffle = False,
        collate_fn = dataset.collate_fn
    )
    for d in data_loader:
        test_X = tf.convert_to_tensor(d['encoded_tokens'], dtype=tf.float32)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SeqClassifier(
        embeddings=embeddings,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_class=dataset.num_classes,
    )
    x = tf.ones((128, 36))
    y = tf.ones((128, 36))
    model(x, y, False, None, None, None)

    # load weights into model
    model.load_weights(args.ckpt_path)

    # TODO: predict dataset
    pred_Y = test_step(model, test_X, args.max_len)

    print(pred_Y)
    print(pred_Y[0])

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'slot'])
        for i, row in enumerate(pred_Y):
            pred_Y_labels = []
            for idx in row:
                pred_Y_labels.append(dataset.idx2label(int(idx)))
            writer.writerow([f"test-{i}", ' '.join(pred_Y_labels)])
    return


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
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=37)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
