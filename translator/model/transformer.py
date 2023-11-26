import torch
import torch.nn as nn
import lightning as L
import torchmetrics as tm


class TransformerSeq2Seq(L.LightningModule):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        lr=1e-3,
        weight_decay=1e-2,
        sos_idx=1,
        eos_idx=2,
        padding_idx=3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.fc = nn.Linear(embedding_dim, tgt_vocab_size)

        self.criteria = nn.CrossEntropyLoss()

    def forward(self, src, tgt):
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        out = self.transformer(src, tgt)
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        logits = self(src, tgt)
        logits = logits.permute(0, 2, 1)
        loss = self.criteria(logits, tgt)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        logits = self(src, tgt)
        logits = logits.permute(0, 2, 1)
        loss = self.criteria(logits, tgt)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
