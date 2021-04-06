''' Libraries from sample code '''
from typing import List, Dict
from torch.utils.data import Dataset
from utils import Vocab


''' Libraries added by me '''
import torch.tensor as tensor
from torch.nn.functional import one_hot


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn  # Done
        split_text = [ s['text'].split() for s in samples ]
        output = {
            # 'text': [ s['text'] for s in samples ],
            # 'split_text': split_text,
            # 'split_text_no_mark': [ s['text'].replace(',', '').replace('!', '').split() for s in samples ],
            # 'intent': [ s['intent'] for s in samples ],
            # 'intent_idx': [ self.label2idx(s['intent']) for s in samples ],
            # 'split_intent': [ s['intent'].split('_') for s in samples ],
            # 'id': [ s['id'] for s in samples ],
            # 'id': [ int(s['id'].split('-')[1]) for s in samples ],
        }
        output['embedded_split_text'] = self.vocab.encode_batch(split_text, to_len=self.max_len)
        output['intent_one_hot'] = [
            one_hot(tensor(self.label2idx(s['intent'])), self.num_classes) for s in samples
        ]
        output['intent_one_hot'] = [ l.numpy().tolist() for l in output['intent_one_hot'] ]
        return output
        # raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]