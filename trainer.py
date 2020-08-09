# -*- coding: utf-8 -*-

from torch.nn import CrossEntropyLoss, NLLLoss
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import train_test_split

from src.info import modelInfo
from src.datagen import Datagen
from src.utils import Collate
from src.utils import getDevice
from src.architecture import CNNtoRNN
from src.train import train_model
from src.dataclass import *

def main(
    LoaderParams: dc_LoaderParams,
    ModelParams: dc_ModelParams,
    TrainParams: dc_TrainParams):

    # LOADER

    train_transform = transforms.Compose(
        [
            transforms.Resize(LoaderParams.image_shape),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    validation_transforms = transforms.Compose(
        [
            transforms.Resize(LoaderParams.image_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_set = Datagen(
                root_dir=LoaderParams.root_folder,
                captions_file=LoaderParams.annotation_file,
                transform=train_transform)

    validation_set = Datagen(
                    root_dir=LoaderParams.root_folder,
                    captions_file=LoaderParams.annotation_file,
                    transform=validation_transforms)

    train_idx, val_idx = train_test_split(list(range(len(train_set.df))), test_size=LoaderParams.val_split)  #type:ignore
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)

    pad_idx = train_set.vocab.stoi["<PAD>"]
    train_loader = DataLoader(
        dataset=train_set,
        sampler=train_sampler,
        batch_size=LoaderParams.batch_size,
        num_workers=LoaderParams.num_workers,
        pin_memory=LoaderParams.pin_memory,
        collate_fn=Collate(pad_idx=pad_idx),
    )
    valid_loader = DataLoader(
        dataset=validation_set,
        sampler=valid_sampler,
        batch_size=LoaderParams.batch_size,
        num_workers=LoaderParams.num_workers,
        pin_memory=LoaderParams.pin_memory,
        collate_fn=Collate(pad_idx=pad_idx),
    )

    dataloaders = {"train": train_loader, "val": valid_loader}
    dataloader_len = {"train": len(train_idx), "val": len(val_idx)}
    vocab_size = len(train_set.vocab)

    # MODEL

    device = getDevice()
    model = CNNtoRNN(
                embed_size=ModelParams.embed_size,
                hidden_size=ModelParams.hidden_size,
                vocab_size=vocab_size,
                num_layers=ModelParams.num_layers,
                device=device,
                cnn_train_base=False,
                cnn_fine_tune=True).to(device)

    # TRAIN

    criterion = CrossEntropyLoss(ignore_index=train_set.vocab.stoi["<PAD>"])
    # criterion = NLLLoss(ignore_index=train_set.vocab.stoi["<PAD>"])
    optimizer = Adam(model.parameters(), lr=TrainParams.learning_rate)
    nepochs = TrainParams.num_epochs

    train_model(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        dataloaders=dataloaders,
        dataloader_len=dataloader_len,
        input_shape=None,
        scheduler=None,
        num_epochs=nepochs)


if __name__ == "__main__":

    LoaderParams = dc_LoaderParams(
        root_folder = "dataset/images",
        annotation_file = "dataset/captions.csv",
        image_shape = (256, 256)
        )

    ModelParams = dc_ModelParams(
        embed_size = 256,
        hidden_size = 256,
        num_layers = 2
        )

    TrainParams = dc_TrainParams(
        input_shape = (3, 256, 256),
        num_epochs = 50
    )

    main(LoaderParams, ModelParams, TrainParams)
