import sys

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from data.opus.datamodule import OPUS100DataModule
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

    wandb_logger = WandbLogger(
        project="machine-translation",
    )
    lr_monitor = LearningRateMonitor(
        logging_interval="step",
        log_momentum=True,
    )
    trainer = L.Trainer(
        max_epochs=10,
        val_check_interval=0.25,
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=[
            lr_monitor,
        ],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    train(
        sys.argv[1],
        sys.argv[2],
    )
