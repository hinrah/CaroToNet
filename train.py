import math
import os

import argparse
from torch.nn import BCEWithLogitsLoss
import torch
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import monai.transforms as mt
from monai.data import CacheDataset
import monai.data
from monai.networks.layers.factories import Norm
import numpy as np

from constants import INIT_LR, NUM_EPOCHS, BATCH_SIZE, SLICES, IMAGE_KEY, LABEL_KEY, KEYS
from helpers import make_sure_folder_exists, search_for_data
from constants import INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, DEVICE
from trainer import Trainer

TRAINING_TRANSFORM = mt.Compose(
    [
        mt.LoadImageD(KEYS, reader="monai.data.NibabelReader", ensure_channel_first=True),
        mt.SqueezeDimd(KEYS, dim=-1),
        mt.SqueezeDimd(KEYS, dim=-1),
        mt.SqueezeDimd(KEYS, dim=-1),
        mt.ToDeviced(KEYS, DEVICE),
        mt.AsDiscreted(LABEL_KEY, to_onehot=3),
        mt.NormalizeIntensityd(IMAGE_KEY, channel_wise=True),
        mt.RandAffineD(
            KEYS,
            rotate_range=(np.pi, 0, 0),
            translate_range=(5, 5, 0),
            scale_range=(0.1, 0.1, 0),
            prob=0.5,
        )
    ]
)

VALIDATION_TRANSFORM = mt.Compose(
    [
        mt.LoadImageD(KEYS, reader="monai.data.NibabelReader", ensure_channel_first=True),
        mt.SqueezeDimd(KEYS, dim=-1),
        mt.SqueezeDimd(KEYS, dim=-1),
        mt.SqueezeDimd(KEYS, dim=-1),
        mt.ToDeviced(KEYS, DEVICE),
        mt.AsDiscreted(LABEL_KEY, to_onehot=3),
        mt.NormalizeIntensityd(IMAGE_KEY, channel_wise=True)
    ]
)

class DataHandler:
    def __init__(self, data_path, output_path, left_out_slice):
        self._data_path = data_path
        self._output_path = output_path
        self._left_out_slice = left_out_slice
        self._train_loader = None
        self._validation_loader = None

        output_folder = os.path.join(output_path, f"slice_{left_out_slice}")
        make_sure_folder_exists(output_folder)
        self.model_path = os.path.join(output_folder, "CaroToNet.pth")
        self.plot_path = os.path.sep.join([output_folder, "plot.png"])

    @property
    def validation_loader(self):
        return self._validation_loader

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def num_validation_samples(self):
        return len(self.validation_loader.dataset)

    @property
    def num_train_samples(self):
        return len(self.train_loader.dataset)

    def initialize(self):
        X_paths, y_paths = search_for_data(self._data_path, left_out_slice=self._left_out_slice)

        train_images = [paths[0] for paths in X_paths]
        train_masks = [path for path in y_paths]

        train_files = [{IMAGE_KEY: img, LABEL_KEY: mask} for img, mask in zip(train_images, train_masks)]

        train_data_set = CacheDataset(data=train_files, transform=TRAINING_TRANSFORM, cache_rate=1.0,
                                      num_workers=8)

        self._train_loader = monai.data.DataLoader(train_data_set, shuffle=True, batch_size=BATCH_SIZE)




def run_training(data_path, output_path):
    for left_out_slice in SLICES:
        data_handler = DataHandler(data_path, output_path, left_out_slice)
        data_handler.initialize()

        unet = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=3,
            dropout=0.1,
            norm=Norm.BATCH
        ).to(DEVICE)
        lossFunc = BCEWithLogitsLoss()
        opt = Adam(unet.parameters(), lr=INIT_LR)
        trainer = Trainer(unet, opt, lossFunc, data_handler.train_loader, None)
        for e in tqdm(range(NUM_EPOCHS)):
            _ = trainer.train_step()
            torch.save(unet, data_handler.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train 2D vessel segmentation',
                                     description='This trains the 2D vessel segmenation. The training trains 9 models. One with all training data and 8 models '
                                                 'that do not contain the cross-sections at one plane.')
    parser.add_argument('-d', '--data_path')
    parser.add_argument('-o', '--output_path')
    args = parser.parse_args()

    run_training(args.data_path, args.output_path)
