# -*- coding: utf-8 -*-

import torch

def testModel(model, testloader, device, encoder=None):
    raise NotImplementedError


def testImage(model, image, vocabulary, device, max_length=50):
    result_caption = []

    with torch.no_grad():
        x = model.encoderCNN(image.to(device)).unsqueeze(0)
        states = None

        for _ in range(max_length):
            hiddens, states = model.decoderRNN.lstm(x, states)
            output = model.decoderRNN.linear(hiddens.squeeze(0))
            predicted = output.argmax(1)
            result_caption.append(predicted.item())
            x = model.decoderRNN.embed(predicted).unsqueeze(0)

            if vocabulary.itos[predicted.item()] == "<EOS>":
                break

    return [vocabulary.itos[idx] for idx in result_caption]