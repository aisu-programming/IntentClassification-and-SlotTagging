''' Libraries from sample code '''
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import torch
from tqdm import trange
from dataset import SeqClsDataset
from utils import Vocab


''' Libraries added by me '''
import os, time
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from torch.utils.data import DataLoader
# from pytorch_model import SeqClassifier
from tf_model_2 import SeqClassifier


''' Parameters from sample code '''
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


''' Functions '''
def parse_args() -> Namespace:

    now_time = time.strftime('%m%d_%H%M%S', time.localtime())

    parser = ArgumentParser()

    # data
    parser.add_argument("--max_len", type=int, default=35)

    # model
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--decay_rate", type=float, default=0.9)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device,
        help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--restore_best_weights", type=bool, default=False)

    # path
    parser.add_argument(
        "--data_dir", type=Path, help="Directory to the dataset.",
        default="./data/slot")
    parser.add_argument(
        "--cache_dir", type=Path, help="Directory to the preprocessed caches.",
        default="./cache/slot")
    parser.add_argument(
        "--ckpt_dir", type=Path, help="Directory to save the model file.",
        default=f"./ckpt/slot/{now_time}")
    parser.add_argument(
        "--logs_dir", type=Path, help="Directory to save the logs file.",
        default=f"./logs/slot/{now_time}")

    args = parser.parse_args()
    return args


def save_args(args):
    d = vars(args)
    with open(f"{args.ckpt_dir}/args.txt", mode='w') as f:
        for key in d.keys():
            f.write(f"{key:20}: {d[key]}\n")
    return
    

def __get_data(cache_dir) -> Dict[str, DataLoader]:

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets  # Done
    data_loaders: Dict[str, DataLoader] = {
        split: DataLoader(
            split_dataset,
            batch_size = 7244,
            shuffle = False,
            collate_fn = split_dataset.collate_fn_slot
        ) for split, split_dataset in datasets.items()
    }

    return datasets, data_loaders


def main(args):
    
    datasets, data_loaders = __get_data(args.cache_dir)

    for d in data_loaders[TRAIN]:
        train_X = tf.convert_to_tensor(d['encoded_tokens'], dtype=tf.float32)
        train_Y = tf.convert_to_tensor(d['tag_one_hot'], dtype=tf.float32)

    for d in data_loaders[DEV]:
        valid_X = tf.convert_to_tensor(d['encoded_tokens'], dtype=tf.float32)
        valid_Y = tf.convert_to_tensor(d['tag_one_hot'], dtype=tf.float32)

    embeddings = tf.convert_to_tensor(torch.load(args.cache_dir / "embeddings.pt"))
    model = SeqClassifier(
        embeddings=embeddings,
        text_len=args.max_len,       # Added by me
        batch_size=args.batch_size,  # Added by me
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=datasets[TRAIN].num_classes,
        mode='slot'                  # Added by me
    )
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.lr,
        decay_steps=10000,
        decay_rate=args.decay_rate
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        # run_eagerly=True
    )

    callbacks = [
        # MCP_ValLoss
        tf.keras.callbacks.ModelCheckpoint(
            f"{args.ckpt_dir}/MinValLoss.h5",
            monitor='val_loss', mode='min', verbose=1,
            save_best_only=True # , save_weights_only=True
        ),
        # MCP_ValAcc
        tf.keras.callbacks.ModelCheckpoint(
            f"{args.ckpt_dir}/MaxValAcc.h5",
            monitor='val_accuracy', mode='max', verbose=1,
            save_best_only=True # , save_weights_only=True
        ),
        # # ES_ValLoss
        # tf.keras.callbacks.EarlyStopping(
        #     monitor='val_loss', mode='min',
        #     verbose=1, patience=args.patience,
        #     restore_best_weights=args.restore_best_weights
        # ),
        # ES_ValAcc
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max',
            verbose=1, patience=args.patience,
            restore_best_weights=args.restore_best_weights
            # baseline=0.85022
        ),
        # TB
        tf.keras.callbacks.TensorBoard(
            log_dir=args.logs_dir,
            write_graph=True, write_images=True,
        )
    ]

    # time.sleep(3)
    os.system('clear')
    os.system('clear')

    # print args
    print("\n")
    d = vars(args)
    for key in d.keys():
        print(f"{key:20}: {d[key]}")
    print("\n")

    history = model.fit(
        x=train_X,
        y=train_Y,
        batch_size=args.batch_size,
        epochs=args.num_epoch,
        verbose=1,
        callbacks=callbacks,
        validation_data=(valid_X, valid_Y),
        shuffle=True,
    )

    Min_ValLoss = min(history.history['val_loss'])
    MVL_ValAcc  = history.history['val_accuracy'][history.history['val_loss'].index(Min_ValLoss)]
    print(f"\n\nMin val_loss: {Min_ValLoss} ({Min_ValLoss:.5f}) --> val_accuracy: {MVL_ValAcc} ({MVL_ValAcc*100:.3f}%)")

    Max_ValAcc  = max(history.history['val_accuracy'])
    MVA_ValLoss = history.history['val_loss'][history.history['val_accuracy'].index(Max_ValAcc)]
    print(f"\n\nMax val_accuracy: {Max_ValAcc} ({Max_ValAcc*100:.3f}%) --> val_loss: {MVA_ValLoss} ({MVA_ValLoss:.5f})")

    print("\n")
    args.ckpt_dir.rename(f"{args.ckpt_dir}_{Min_ValLoss:.5f}-{MVL_ValAcc*100:.3f}%_{Max_ValAcc*100:.3f}%-{MVA_ValLoss:.5f}")

    # TODO: Inference on test set
    return


''' Execution '''
if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir = Path(f"{args.ckpt_dir}_numlayers={args.num_layers}")
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir = Path(f"{args.logs_dir}_numlayers={args.num_layers}")
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    save_args(args)
    main(args)