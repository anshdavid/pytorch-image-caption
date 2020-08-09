# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class VGG16Encoder(nn.Module):

    def __init__(self, output, device, train_base=False, fine_tune=True):

        super(VGG16Encoder, self).__init__()

        # self.device = device
        self.output = output
        self.toTrain = train_base
        self.toTune = fine_tune

        self.net_back = models.vgg16(pretrained=True)#.to(self.device)
        self._TrainTune()

        self.net_back.classifier = Identity()
        self.net_head = nn.Sequential(
            nn.Linear(in_features=25088, out_features=1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=self.output)
            )

    def _TrainTune(self):

        for name, param in self.net_back.named_parameters():
            # fine tune !
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = self.toTune
            # is trainable ?
            else:
                param.requires_grad = self.toTrain


    def forward(self, image):

        x = self.net_back(image)#.to(self.device))
        x = x.view(x.size(0), -1)
        return self.net_head(x)#.to(self.device))


class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, lstm_num_layers, device):

        super(LSTMDecoder, self).__init__()

        self.embedSize = embed_size
        self.hiddenSize = hidden_size
        self.vocabSize = vocab_size
        self.numLayers = lstm_num_layers
        # self.device = device

        self.dropout = nn.Dropout(0.5)

        # lookup table
        self.embed = nn.Embedding(
            num_embeddings=self.vocabSize,
            embedding_dim=self.embedSize)

        self.lstm = nn.LSTM(
            input_size=self.embedSize,
            hidden_size=self.hiddenSize,
            num_layers=self.numLayers)

        self.linear = nn.Linear(in_features=self.hiddenSize, out_features=self.vocabSize)

        # self.net_head = nn.Sequential(
        #     nn.Linear(in_features=self.hiddenSize, out_features=self.vocabSize),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        # )


    def forward(self, features, captions):

        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)  #type:ignore
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs


class CNNtoRNN(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        num_layers,
        device,
        cnn_train_base=False,
        cnn_fine_tune=True):

        super(CNNtoRNN, self).__init__()

        # self.device = device

        self.encoderCNN = VGG16Encoder(
            output=embed_size,
            device=device,
            train_base=cnn_train_base,
            fine_tune=cnn_fine_tune)

        self.decoderRNN = LSTMDecoder(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            lstm_num_layers=num_layers,
            device=device)


    def forward(self, images, captions):

        # features = self.encoderCNN(images.to(self.device))
        # outputs = self.decoderRNN(features.to(self.device), captions.to(self.device))
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs
