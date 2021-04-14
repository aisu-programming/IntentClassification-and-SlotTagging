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
        self._idx2label = {idx: label for label, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)


    def collate_fn_intent(self, samples: List[Dict]) -> Dict:
        split_text = [ s['text'].split() for s in samples ]
        output = {
            # 'id': [ s['id'] for s in samples ],
            # 'id': [ int(s['id'].split('-')[1]) for s in samples ],
            'encoded_split_text': self.vocab.encode_batch(split_text, to_len=self.max_len),
        }

        if 'intent' in samples[0].keys():
            output['intent_one_hot'] = [
                one_hot(tensor(self.label2idx(s['intent'])), self.num_classes) for s in samples
            ]
            output['intent_one_hot'] = [ l.numpy().tolist() for l in output['intent_one_hot'] ]
        
        return output


    def collate_fn_slot(self, samples: List[Dict]) -> Dict:
        tokens = [ s['tokens'] for s in samples ]
        # tag_one_hot = [ s['tags'] for s in samples ]
        # for i, row in enumerate(tag_one_hot):
        #     for j in range(len(row)):
        #         tag_one_hot[i][j] = \
        #             one_hot(tensor(self.label2idx(tag_one_hot[i][j])), self.num_classes).numpy().tolist()
        #     if len(row) != self.max_len:
        #         for _ in range(self.max_len-len(row)):
        #             tag_one_hot[i].append([0]*self.num_classes)
        output = {
            # 'id': [ s['id'] for s in samples ],
            # 'id': [ int(s['id'].split('-')[1]) for s in samples ],
            'encoded_tokens': self.vocab.encode_batch(tokens, to_len=self.max_len),
            # 'tag_one_hot': tag_one_hot,
        }

        if 'tags' in samples[0].keys():
            tags = [ s['tags'] for s in samples ]
            for i, tag_row in enumerate(tags):
                tags_i_tmp = ['S']
                for tag in tag_row: tags_i_tmp.append(tag)
                for _ in range(self.max_len-len(tags_i_tmp)-1): tags_i_tmp.append('P')
                tags_i_tmp.append('E')
                tags[i] = tags_i_tmp
                for j in range(len(tag_row)):
                    tags[i][j] = self.label2idx(tags[i][j])
            output['tags'] = tags
        
        return output


    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]