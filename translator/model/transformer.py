import torch
import torch.nn as nn
import lightning as L
import math

from utils import generate_square_subsequent_mask


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(
            -torch.arange(0, embedding_dim, 2) * math.log(10000) / embedding_dim
        )
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, embedding_dim))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TransformerSeq2Seq(L.LightningModule):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embedding_dim=512,
        hidden_dim=512,
        dropout=0.1,
        nhead=8,
        num_layers=3,
        lr=1e-4,
        weight_decay=1e-2,
        sos_idx=1,
        eos_idx=2,
        padding_idx=3,
    ):
        super().__init__()
        self.save_hyperparameters()

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
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )
        self.fc = nn.Linear(embedding_dim, tgt_vocab_size)

        self.criteria = nn.CrossEntropyLoss()

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
    ):
        src = self.src_embedding(src) * (self.hparams.embedding_dim**0.5)
        tgt = self.tgt_embedding(tgt) * (self.hparams.embedding_dim**0.5)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        out = self.transformer(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        out = self.fc(out)
        return out

    def greedy_decode(self, src, max_len):
        src = self.src_embedding(src) * (self.hparams.embedding_dim**0.5)
        src = self.positional_encoding(src)
        memory = self.transformer.encoder(src)
        ys = torch.ones(1, 1).fill_(self.hparams.sos_idx).type(torch.long)
        for i in range(max_len - 1):
            tgt = self.tgt_embedding(ys) * (self.hparams.embedding_dim**0.5)
            tgt = self.positional_encoding(tgt)
            tgt_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool)
            out = self.transformer.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
            )
            out = self.fc(out)
            out = out.transpose(0, 1)[:, -1]
            prob = out.softmax(dim=-1)
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat(
                [ys, torch.ones(1, 1).fill_(next_word).type(torch.long)],
                dim=0,
            )

            if next_word == self.hparams.eos_idx:
                break

        return ys

    def training_step(self, batch, batch_idx):
        src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = batch
        tgt_input = tgt[:-1, :]
        logits = self(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
        )
        tgt_out = tgt[1:, :]
        loss = self.criteria(
            logits.reshape(-1, logits.shape[-1]),
            tgt_out.reshape(-1),
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = batch
        tgt_input = tgt[:-1, :]
        logits = self(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
        )
        tgt_out = tgt[1:, :]
        loss = self.criteria(
            logits.reshape(-1, logits.shape[-1]),
            tgt_out.reshape(-1),
        )
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
