import torch
from torch.utils.data import DataLoader
import lightning as L
from tokenizers import Tokenizer

from data.opus.dataset import OPUS100Dataset
from utils import generate_square_subsequent_mask


class OPUS100DataModule(L.LightningDataModule):
    def __init__(
        self,
        src_tokenizer_ckpt_path,
        tgt_tokenizer_ckpt_path,
        subset="en-hi",
        batch_size=32,
        num_workers=2,
        sos_idx=1,
        eos_idx=2,
        padding_idx=3,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.subset = subset
        self.src_tokenizer_ckpt_path = src_tokenizer_ckpt_path
        self.tgt_tokenizer_ckpt_path = tgt_tokenizer_ckpt_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download, tokenize, etc...
        self.train_ds = OPUS100Dataset(subset=self.subset, split="train")
        self.val_ds = OPUS100Dataset(subset=self.subset, split="validation")
        self.test_ds = OPUS100Dataset(subset=self.subset, split="test")
        self.src_tokenizer = Tokenizer.from_file(self.src_tokenizer_ckpt_path)
        self.tgt_tokenizer = Tokenizer.from_file(self.tgt_tokenizer_ckpt_path)

    def setup(self, stage):
        pass

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.float32)

        src_padding_mask = (
            (src == self.hparams.padding_idx).transpose(0, 1).type(torch.float32)
        )
        tgt_padding_mask = (
            (tgt == self.hparams.padding_idx).transpose(0, 1).type(torch.float32)
        )
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def collate_fn(self, batch):
        src, tgt = zip(*batch)
        src_encodings = self.src_tokenizer.encode_batch(src)
        tgt_encodings = self.tgt_tokenizer.encode_batch(tgt)
        src_ids = [enc.ids for enc in src_encodings]
        tgt_ids = [enc.ids for enc in tgt_encodings]
        src = torch.LongTensor(src_ids).permute(1, 0)
        tgt = torch.LongTensor(tgt_ids).permute(1, 0)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(
            src, tgt[:-1, :]
        )
        return (
            src,
            tgt,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
        )

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
