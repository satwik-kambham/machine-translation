import torch
import torch.nn as nn
import lightning as L
import torchmetrics as tm


class LSTMSeq2Seq(L.LightningModule):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_embedding_dim=256,
        tgt_embedding_dim=256,
        hidden_dim=256,
        num_layers=1,
        dropout=0.0,
        lr=1e-3,
        weight_decay=1e-2,
        sos_idx=1,
        padding_idx=3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.sos_idx = sos_idx
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            src_embedding_dim,
            padding_idx=padding_idx,
        )
        self.encoder_lstm = nn.LSTM(
            src_embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            tgt_embedding_dim,
            padding_idx=padding_idx,
        )
        self.decoder_lstm = nn.LSTM(
            tgt_embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

        self.criteria = nn.CrossEntropyLoss()

    def forward(self, src, tgt_len, tgt=None):
        x = self.src_embedding(src)
        x, (h, c) = self.encoder_lstm(x)

        decoder_input = torch.full(
            (src.size(0), 1),
            self.sos_idx,
            dtype=torch.long,
        )
        outputs = []

        for i in range(tgt_len):
            x = self.tgt_embedding(decoder_input)
            x, (h, c) = self.decoder_lstm(x, (h, c))
            x = self.fc(x)
            outputs.append(x)
            if self.training:
                decoder_input = tgt[:, i].unsqueeze(1)
            else:
                decoder_input = x.argmax(dim=-1)

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def training_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        logits = self(src, tgt.size(1), tgt)
        logits = logits.permute(0, 2, 1)
        loss = self.criteria(logits, tgt)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        logits = self(src, tgt.size(1), tgt)
        logits = logits.permute(0, 2, 1)
        loss = self.criteria(logits, tgt)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
