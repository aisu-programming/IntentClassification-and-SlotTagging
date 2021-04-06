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
from tf_model import SeqClassifier


''' Parameters from sample code '''
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


''' Functions '''
def parse_args() -> Namespace:

    now_time = time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime())

    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default=f"./ckpt/intent/{now_time}",
    )
    parser.add_argument(
        "--logs_dir",
        type=Path,
        help="Directory to save the logs file.",
        default=f"./logs/intent/{now_time}",
    )

    # data
    parser.add_argument("--max_len", type=int, default=30)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)

    args = parser.parse_args()
    return args


def save_args(args):
    d = vars(args)
    with open(f"{args.ckpt_dir}/args.txt", mode='w') as f:
        for key in d.keys():
            f.write(f"{key:13}: {d[key]}\n")
    return
    

def __get_data(cache_dir) -> Dict[str, DataLoader]:

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets  # Done
    data_loaders: Dict[str, DataLoader] = {
        split: DataLoader(
            split_dataset,
            # batch_size = args.batch_size,
            batch_size = 15000,
            shuffle = False,
            collate_fn = split_dataset.collate_fn
        ) for split, split_dataset in datasets.items()
    }

    return datasets, data_loaders


def main(args):
    
    datasets, data_loaders = __get_data(args.cache_dir)

    for d in data_loaders[TRAIN]:
        train_Y = tf.convert_to_tensor(d['intent_one_hot'], dtype=tf.float32)
        train_X = tf.convert_to_tensor(d['embedded_split_text'], dtype=tf.float32)

    for d in data_loaders[DEV]:
        valid_X = tf.convert_to_tensor(d['embedded_split_text'], dtype=tf.float32)
        valid_Y = tf.convert_to_tensor(d['intent_one_hot'], dtype=tf.float32)

    # TODO: init model and move model to target device(cpu / gpu)
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
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True
    )

    callbacks = [
        # MCP_ValLoss
        tf.keras.callbacks.ModelCheckpoint(
            f"{args.ckpt_dir}/MinValLoss.h5",
            monitor='val_loss', mode='min', verbose=1,
            save_best_only=True, save_weights_only=True
        ),
        # # MCP_ValAcc
        # tf.keras.callbacks.ModelCheckpoint(
        #     f"{args.ckpt_dir}/MaxValAcc.h5",
        #     monitor='val_accuracy', mode='max', verbose=1,
        #     save_best_only=True, save_weights_only=True
        # ),
        # ES_ValLoss
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min',
            verbose=1, patience=args.patience
        ),
        # # ES_ValAcc
        # tf.keras.callbacks.EarlyStopping(
        #     monitor='val_accuracy', mode='max',
        #     verbose=1, patience=args.patience
        # ),
        # TB
        tf.keras.callbacks.TensorBoard(
            log_dir=args.logs_dir,
            write_graph=True,
            write_images=True,
        )
    ]

    # time.sleep(3)
    os.system('clear')
    os.system('clear')

    # print args
    print("\n\n")
    d = vars(args)
    for key in d.keys():
        print(f"{key:13}: {d[key]}")
    print("\n\n")

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

    results = model.evaluate(
        x=valid_X,
        y=valid_Y,
        batch_size=args.batch_size,
    )

    print(f"\n\nMin val_loss: {min(history.history['val_loss'])}\n\n")
    args.ckpt_dir.rename(f"{args.ckpt_dir}_{min(history.history['val_loss'])}")

    '''
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    criterion = tf.keras.losses.CategoricalCrossentropy()

    for i in range(args.num_epoch):
        print(f"Epoch: {i+1:2d}/{args.num_epoch:2d} | ", end='')
        average_loss = []
        for j in range(int(15000/args.batch_size)):
            
            if j == int(15000/args.batch_size):
                batch_X = train_X[j*args.batch_size:]
                batch_Y = train_Y[j*args.batch_size:]
            else:
                batch_X = train_X[j*args.batch_size:(j+1)*args.batch_size]
                batch_Y = train_Y[j*args.batch_size:(j+1)*args.batch_size]
            
            # pred_Y = model.call(batch_X)
            # train  = optimizer.minimize(loss_func(batch_Y, pred_Y), [pred_Y])

            with tf.GradientTape() as tape:
                pred_Y = model.call(batch_X)
                loss = criterion(batch_Y, pred_Y)
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

            average_loss.append(loss)
        
        average_loss = np.average(average_loss)
        print(average_loss)

        model.save(
            f"{args.ckpt_dir}test.h5",
        )
    '''


    # TODO: Inference on test set
    return


''' Execution '''
if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    save_args(args)
    main(args)