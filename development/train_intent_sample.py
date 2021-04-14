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
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.tensor as tensor
from tqdm import tqdm
from torch.utils.data import DataLoader
# from pytorch_model import SeqClassifier
from tf_model import SeqClassifier


''' Parameters from sample code '''
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


''' Functions '''
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
            batch_size = args.batch_size,
            shuffle = True if split == TRAIN else False,
            collate_fn = split_dataset.collate_fn
        ) for split, split_dataset in datasets.items()
    }

    return datasets, data_loaders


def __select_optimizer(model):
    # TODO: init optimizer  # Done
    optimizer = optim.Adam(model.parameters())
    return optimizer


def __select_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion


def calculate_accuracy(one_hot, answer):
    accuracy = []
    for i, row in enumerate(one_hot.numpy()):
        accuracy.append(row[answer[i]])
    accuracy = np.average(accuracy)
    return accuracy


def main(args):
    
    datasets, data_loaders = __get_data(args.cache_dir)

    # print(datasets[DEV].num_classes)
    # for i in data_loaders[TRAIN]:
    #     print(i)
    # return

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings=embeddings,
        em_zise=args.max_len,  # Added by me
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=datasets[TRAIN].num_classes,
    )
    model = model.to(args.device)

    optimizer = __select_optimizer(model)
    criterion = __select_criterion()

    # epoch_pbar = trange(, desc="Epoch")
    best_accuracy = 0.
    best_loss = np.inf
    for epoch in range(args.num_epoch):
        # TODO: Training loop - iterate over train dataloader and update model weights
        average_loss = []
        average_accuracy = []
        for dictionary in tqdm(data_loaders[TRAIN], desc=f"Epoch: {epoch+1:2d}/{args.num_epoch}"):

            optimizer.zero_grad()

            embedded_split_text = tensor(dictionary['embedded_split_text'], device=args.device)
            intent_idx          = tensor(dictionary['intent_idx'], device=args.device)
            intent_one_hot      = tensor(dictionary['intent_one_hot'], device=args.device)

            output = model.forward(embedded_split_text, intent_one_hot)
            output = output.detach()

            accuracy = calculate_accuracy(output, intent_idx)
            average_accuracy.append(accuracy)
            loss = criterion(output.requires_grad_(), intent_idx)
            average_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()

        average_accuracy = np.average(average_accuracy)
        if average_accuracy > best_accuracy: 
            print(f"Accuracy: {average_accuracy * 100:.5f}%... Improved!")
            best_accuracy = average_accuracy
        else: print(f"Accuracy: {average_accuracy * 100:.5f}%.")

        average_loss = np.average(average_loss)
        if average_loss < best_loss: 
            print(f"Loss: {average_loss:.10f}... Improved!")
            best_loss = average_loss
        else: print(f"Loss: {average_loss:.10f}.")

        print('')

        # TODO: Evaluation loop - calculate accuracy and save model weights
        pass

    # TODO: Inference on test set
    return


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=30)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=150)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=30)

    args = parser.parse_args()
    return args


''' Execution '''
if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)