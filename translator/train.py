import sys

import lightning as L

from data.opus.datamodule import OPUS100DataModule
from model.lstm import LSTMSeq2Seq
from model.transformer import TransformerSeq2Seq


def train(
    src_tokenizer_ckpt_path,
    tgt_tokenizer_ckpt_path,
):
    L.seed_everything(42)
    dm = OPUS100DataModule(
        src_tokenizer_ckpt_path,
        tgt_tokenizer_ckpt_path,
    )
    dm.prepare_data()
    model = TransformerSeq2Seq(
        dm.src_tokenizer.get_vocab_size(),
        dm.tgt_tokenizer.get_vocab_size(),
    )

    trainer = L.Trainer(
        max_epochs=30,
        fast_dev_run=False,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    train(
        sys.argv[1],
        sys.argv[2],
    )
