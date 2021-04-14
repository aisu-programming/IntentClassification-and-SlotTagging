from typing import Iterable, List


class Vocab:
    PAD = "[PAD]"  # padding
    UNK = "[UNK]"  # unknown
    STA = "[STA]"  # start
    END = "[END]"  # end

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            Vocab.STA: 2,
            Vocab.END: 3,
            **{token: i for i, token in enumerate(vocab, 4)},
        }

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    @property
    def sta_id(self) -> int:
        return self.token2idx[Vocab.STA]

    @property
    def end_id(self) -> int:
        return self.token2idx[Vocab.END]

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [ self.encode(tokens) for tokens in batch_tokens ]
        # for i, ids in enumerate(batch_ids):
        #     batch_ids[i].insert(0, 2)  # START id
        #     batch_ids[i].append(3)     # END   id
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_id)
        return padded_ids


def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [ seq + [padding]*max(0, to_len-len(seq)) for seq in seqs ]
    return paddeds