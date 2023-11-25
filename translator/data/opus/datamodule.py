import torch
from torch.utils.data import DataLoader
import lightning as L
from tokenizers import Tokenizer

from data.opus.dataset import OPUS100Dataset


class OPUS100DataModule(L.LightningDataModule):
    def __init__(
        self,
        src_tokenizer_ckpt_path,
        tgt_tokenizer_ckpt_path,
        subset="en-hi",
        batch_size=32,
        num_workers=2,
        **kwargs,
    ):
        super().__init__()
        self.subset = subset
        self.src_tokenizer_ckpt_path = src_tokenizer_ckpt_path
        self.tgt_tokenizer_ckpt_path = tgt_tokenizer_ckpt_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download, tokenize, etc...
        self.train_ds = OPUS100Dataset(split="train")
        self.val_ds = OPUS100Dataset(split="validation")
        self.test_ds = OPUS100Dataset(split="test")
        self.src_tokenizer = Tokenizer.from_file(self.src_tokenizer_ckpt_path)
        self.tgt_tokenizer = Tokenizer.from_file(self.tgt_tokenizer_ckpt_path)

    def setup(self, stage):
        pass

    def collate_fn(self, batch):
        src, tgt = zip(*batch)
        src_encodings = self.src_tokenizer.encode_batch(src)
        tgt_encodings = self.tgt_tokenizer.encode_batch(tgt)
        src_ids = [enc.ids for enc in src_encodings]
        tgt_ids = [enc.ids for enc in tgt_encodings]
        return {
            "src": torch.LongTensor(src_ids),
            "tgt": torch.LongTensor(tgt_ids),
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
