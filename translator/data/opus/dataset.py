from torch.utils.data import Dataset
from datasets import load_dataset


class OPUS100Dataset(Dataset):
    """
    OPUS100 dataset
    """

    def __init__(self, subset="en-hi", split="train"):
        self.src_lang, self.tgt_lang = subset.split("-")
        self.data = load_dataset("opus100", subset, split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]["translation"][self.src_lang]
        tgt = self.data[idx]["translation"][self.tgt_lang]
        return src, tgt
