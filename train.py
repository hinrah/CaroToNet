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

from constants import INIT_LR, NUM_EPOCHS, BATCH_SIZE, SLICES, IMAGE_KEY, LABEL_KEY, KEYS, SPLIT_IDX
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

DUMMY_IN = torch.randn(1, 2, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT).to(DEVICE)


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

        train_images = [paths for paths in X_paths if os.path.basename(paths[0]) < SPLIT_IDX]
        validation_images = [paths for paths in X_paths if os.path.basename(paths[0]) > SPLIT_IDX]
        train_masks = [path for path in y_paths if os.path.basename(path) < SPLIT_IDX]
        validation_masks = [path for path in y_paths if os.path.basename(path) > SPLIT_IDX]

        train_files = [{IMAGE_KEY: img, LABEL_KEY: mask} for img, mask in zip(train_images, train_masks)]
        val_files = [{IMAGE_KEY: img, LABEL_KEY: mask} for img, mask in zip(validation_images, validation_masks)]

        train_data_set = CacheDataset(data=train_files, transform=TRAINING_TRANSFORM, cache_rate=1.0,
                                      num_workers=8)
        validation_data_set = CacheDataset(data=val_files, transform=VALIDATION_TRANSFORM, cache_rate=1.0,
                                           num_workers=8)
        self._train_loader = monai.data.DataLoader(train_data_set, shuffle=True, batch_size=BATCH_SIZE)
        self._validation_loader = monai.data.DataLoader(validation_data_set, shuffle=False, batch_size=BATCH_SIZE)


def plot_loss(losses, plot_path):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(losses["train_loss"], label="train_loss")
    plt.plot(losses["validation_loss"], label="validation_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(plot_path)


def run_training(data_path, output_path):
    for left_out_slice in SLICES:
        data_handler = DataHandler(data_path, output_path, left_out_slice)
        data_handler.initialize()

        unet = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=2,
            out_channels=3,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=3,
            dropout=0.3,
            norm=Norm.BATCH
        ).to(DEVICE)
        lossFunc = BCEWithLogitsLoss()
        opt = Adam(unet.parameters(), lr=INIT_LR)
        trainer = Trainer(unet, opt, lossFunc, data_handler.train_loader, data_handler.validation_loader)

        runs_without_improvement = 0
        train_steps = data_handler.num_train_samples // BATCH_SIZE
        validation_steps = data_handler.num_validation_samples // BATCH_SIZE
        best_validation_lost = math.inf
        losses = {"train_loss": [], "validation_loss": []}

        for e in tqdm(range(NUM_EPOCHS)):
            total_train_loss = trainer.train_step()
            total_validation_loss = trainer.evaluate_step()

            avg_train_loss = total_train_loss / train_steps
            avg_validation_loss = total_validation_loss / validation_steps

            if best_validation_lost > avg_validation_loss:
                best_validation_lost = avg_validation_loss
                torch.save(unet, data_handler.model_path)
                print("saved")
                runs_without_improvement = 0
            else:
                runs_without_improvement += 1
            if runs_without_improvement > 50:
                break

            losses["train_loss"].append(avg_train_loss.cpu().detach().numpy())
            losses["validation_loss"].append(avg_validation_loss.cpu().detach().numpy())
            print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
            print("Train loss: {}, validation loss: {}".format(avg_train_loss, avg_validation_loss))
        plot_loss(losses, data_handler.plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train 2D vessel segmentation',
                                     description='This trains the 2D vessel segmenation. The training trains 9 models. One with all training data and 8 models '
                                                 'that do not contain the cross-sections at one plane.')
    parser.add_argument('-d', '--data_path')
    parser.add_argument('-o', '--output_path')
    args = parser.parse_args()

    run_training(args.data_path, args.output_path)
