import math
from typing import Dict

import torch
import torch.nn as nn
import torch.tensor as tensor


class PositionalEncoding300k(nn.Module):
    def __init__(self, text_len, dropout=0.1, max_len=3000):
        super(PositionalEncoding300k, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = torch.zeros(max_len, text_len)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, text_len, 2).float() * (-math.log(10000.0) / text_len))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(2) * torch.ones(300)

    def forward(self, x):
        x = x + self.pe.to(x.device)[:x.size(0), :, :]
        return self.dropout(x)


class SeqClassifier(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        text_len: int,  # Added by me
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        # TODO: model architecture
        self.E  = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.PE = PositionalEncoding300k(text_len=text_len)
        self.L1 = nn.Linear(300, 1)
        self.L2 = nn.Linear(30, 150)
        self.S  = nn.Softmax(dim=1)
        self.T  = nn.Transformer()

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, embedded_split_text, intent_one_hot) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = self.E(embedded_split_text)
        x = self.PE(x)
        x = self.L1(x)
        x = x.squeeze(2)
        x = self.L2(x)
        x = self.S(x)

        return x
        # raise NotImplementedError