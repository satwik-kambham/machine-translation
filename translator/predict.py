import sys

import torch
import lightning as L
from tokenizers import Tokenizer

from model.lstm import LSTMSeq2Seq
from model.transformer import TransformerSeq2Seq


def setup(
    src_tokenizer_ckpt_path,
    tgt_tokenizer_ckpt_path,
    model_ckpt_path,
):
    L.seed_everything(42)

    src_tokenizer = Tokenizer.from_file(src_tokenizer_ckpt_path)
    tgt_tokenizer = Tokenizer.from_file(tgt_tokenizer_ckpt_path)

    model = TransformerSeq2Seq.load_from_checkpoint(
        model_ckpt_path,
        map_location="cpu",
    )
    model = model.eval()

    return src_tokenizer, tgt_tokenizer, model


if __name__ == "__main__":
    src_tokenizer, tgt_tokenizer, model = setup(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
    )

    while True:
        src = input("Enter source text: ")
        src = src_tokenizer.encode(src).ids
        src = torch.tensor(src).unsqueeze(0)

        tgt = torch.tensor([model.sos_idx]).unsqueeze(0)

        for _ in range(100):
            out = model(src, tgt)
            out = out.argmax(dim=-1)
            out = out[:, -1].unsqueeze(0)
            if out[0, -1] == model.eos_idx:
                break
            tgt = torch.cat([tgt, out], dim=-1)

        tgt = tgt.squeeze(0).tolist()
        print(f"Translated ids: {tgt}")
        tgt = tgt_tokenizer.decode(tgt)
        print(f"Translated text: {tgt}")
