import sys

import torch
import lightning as L
from tokenizers import Tokenizer

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
        tokenized_text = src_tokenizer.encode(src)
        src = torch.LongTensor(tokenized_text.ids).view(-1, 1)
        tgt = model.greedy_decode(src, max_len=100)
        tgt = tgt.squeeze(1).tolist()
        tgt_text = tgt_tokenizer.decode(tgt)
        print(f"Pred: {tgt_text}")
