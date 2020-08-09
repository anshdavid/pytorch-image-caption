# -*- coding: utf-8 -*-

import torch
import torch.cuda as cuda
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)               #type:ignore
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def getDevice():
    if cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


# def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
#     print("=> Saving checkpoint")
#     torch.save(state, filename)


# def load_checkpoint(checkpoint, model, optimizer):
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint["state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer"])
#     step = checkpoint["step"]
#     return step

# def caption_image(self, image, vocabulary, max_length=50):
#     result_caption = []

#     with torch.no_grad():
#         x = self.encoderCNN(image).unsqueeze(0)
#         states = None

#         for _ in range(max_length):
#             hiddens, states = self.decoderRNN.lstm(x, states)
#             output = self.decoderRNN.linear(hiddens.squeeze(0))
#             predicted = output.argmax(1)
#             result_caption.append(predicted.item())
#             x = self.decoderRNN.embed(predicted).unsqueeze(0)

#             if vocabulary.itos[predicted.item()] == "<EOS>":
#                 break

#     return [vocabulary.itos[idx] for idx in result_caption]