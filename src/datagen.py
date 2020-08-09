# -*- coding: utf-8 -*-

import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from src.vocab import Vocabulary

class Datagen(Dataset):
    def __init__(
        self,
        root_dir,
        captions_file,
        transform=None,
        freq_threshold=5):

        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        self.images = self.df["image"]                          #type:ignore
        self.labels = self.df["caption"]                        #type:ignore

        # build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.labels.tolist())

    def __len__(self):
        return len(self.df)                                     #type:ignore

    def __getitem__(self, index):

        caption = self.labels[index]
        img_id = self.images[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)         #type:ignore


