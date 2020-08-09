# -*- coding: utf-8 -*-

from dataclasses import dataclass

@dataclass
class dc_LoaderParams:
    root_folder: str = ""
    annotation_file: str = ""
    image_shape: tuple = tuple()
    num_workers: int = 4
    val_split: float = 0.25
    batch_size: int = 32
    pin_memory: bool = True
    shuffle: bool = True

@dataclass
class dc_ModelParams:
    embed_size: int = int()
    hidden_size: int = int()
    num_layers: int = int()

@dataclass
class dc_TrainParams:
    input_shape: tuple = tuple() #channel first
    learning_rate: float = 3e-4
    num_epochs: int = 100