# -*- coding: utf-8 -*-

import os
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image

from src.dataclass import dc_ModelParams
from src.utils import getDevice
from src.architecture import CNNtoRNN
from src.vocab import Vocabulary
from src.test import testImage


if __name__ == "__main__":

    sampleDir = "./samples"
    validImages = [".jpg",".png", "jpeg"]

    fileCaption = "dataset/captions.csv"

    VocanFreqThreshold = 5
    imageShape = (256, 256)

    device = getDevice()

    ModelParams = dc_ModelParams(
            embed_size = 256,
            hidden_size = 256,
            num_layers = 2
    )

    transform = transforms.Compose([
        transforms.Resize(imageShape),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    vocabDf = pd.read_csv(fileCaption)
    captions = vocabDf["caption"]

    vocab = Vocabulary(VocanFreqThreshold)
    vocab.build_vocabulary(captions.tolist())
    vocabSize = len(vocab)

    model = CNNtoRNN(
                embed_size=ModelParams.embed_size,
                hidden_size=ModelParams.hidden_size,
                vocab_size=vocabSize,
                num_layers=ModelParams.num_layers,
                device=device,
                cnn_train_base=False).to(device)

    model.load_state_dict(torch.load("./models/checkpoint.pth")["state_dict"])
    model.eval()


    for f in os.listdir(sampleDir):

        ext = os.path.splitext(f)[1]

        if ext.lower() not in validImages:
            continue

        imagePath = os.path.join(sampleDir,f)
        image = Image.open(imagePath).convert("RGB")
        tranImage = transform(image).unsqueeze(0)       #type:ignore

        output = testImage(
            model,
            tranImage,
            vocab,
            device = getDevice(),
            max_length=50
        )
        print(f"--- {imagePath}\n{output}\n")
