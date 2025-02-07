import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Optional


class CSNWatermarkingCollator:
    def __init__(self,
                 padding_idx: int,
                 n_bits: int,
                 max_token_len: Optional[int] = 512,
                 require_masks: bool = False) -> None:
        self.padding_idx = padding_idx
        self.n_bits = n_bits
        self.max_token_len = max_token_len
        self.require_masks = require_masks

    def __call__(self, batch):
        token_ids = []
        lengths = []
        watermarks = []

        for tid in batch:
            if self.max_token_len is not None:
                tid = tid[:self.max_token_len]

            token_ids.append(torch.tensor(tid, dtype=torch.long))
            lengths.append(len(tid))
            wm = torch.randint(0, 2, (self.n_bits, ), dtype=torch.long)
            watermarks.append(wm)

        # L, B
        padded_src = pad_sequence(token_ids, padding_value=self.padding_idx)

        if self.require_masks:
            src_padding_mask = (padded_src == self.padding_idx).transpose(0, 1)
            return (
                padded_src,
                torch.tensor(lengths, dtype=torch.long),
                torch.stack(watermarks, dim=0),
                src_padding_mask,
            )
        else:
            return (
                padded_src,
                torch.tensor(lengths, dtype=torch.long),
                torch.stack(watermarks, dim=0),
            )
