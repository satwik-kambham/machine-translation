import sys

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace

from datasets import load_dataset


def setup_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        show_progress=True,
    )

    return tokenizer, trainer


def train_tokenizer_opus100(subset="en-hi"):
    dataset = load_dataset("opus100", subset, split="train")

    def batch_iterator(lang, batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]["translation"]
            batch = [b[lang] for b in batch]
            yield batch

    for lang in subset.split("-"):
        tokenizer, trainer = setup_tokenizer()
        tokenizer.train_from_iterator(
            batch_iterator(lang=lang),
            trainer=trainer,
            length=len(dataset),
        )
        tokenizer.save(f"tokenizer-{lang}.json")


if __name__ == "__main__":
    train_tokenizer_opus100(sys.argv[1])
