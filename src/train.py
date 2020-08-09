# -*- coding: utf-8 -*-

import math
from time import time
from copy import deepcopy

import torch


def train_model(
    model,
    device,
    criterion,
    optimizer,
    dataloaders,
    dataloader_len,
    input_shape=None,
    scheduler=None,
    num_epochs=50,
):

    start = time()
    best_model_wts = deepcopy(model.state_dict())
    best_loss = 0.0

    for epoch in range(num_epochs):
        t_epoch = time()
        print(f"epoch: {epoch+1}/{num_epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for idx, (image, captions) in enumerate(dataloaders[phase]):
                iter_batch = math.ceil(
                    dataloader_len[phase] / dataloaders[phase].batch_size
                )
                print(f"[phase: {phase}] batch: {idx+1}/{iter_batch}", end="\r")

                imgs = image.to(device)
                cptns = captions.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(imgs, cptns[:-1])
                    loss = criterion(
                        outputs.reshape(-1, outputs.shape[2]), 
                        cptns.reshape(-1)
                    )

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * imgs.size(0)

            if phase == "train":
                if scheduler:
                    scheduler.step()

            epoch_loss = running_loss / dataloader_len[phase]
            print(f"[phase: {phase}] Loss: {epoch_loss:.4f}")

            if (phase == "val" and best_loss == 0):
                print(f"[saving model] epoch: {epoch+1}")
                best_loss = epoch_loss
                best_model_wts = deepcopy(model.state_dict())
            elif (phase == "val" and epoch_loss < best_loss):
                print(f"[saving model] epoch: {epoch+1}")
                best_loss = epoch_loss
                best_model_wts = deepcopy(model.state_dict())

        t_elapsed = time() - t_epoch
        print(f"epoch training complete in {t_elapsed//60:.0f}m {t_elapsed%60:.4f}s")
        print()

    time_elapsed = time() - start
    print(f"training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.4f}s")

    model.load_state_dict(best_model_wts)

    if input_shape:
        checkpoint = {
            "input_shape": input_shape,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
    else:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }


    torch.save(checkpoint, "./models/checkpoint.pth")
    return model
