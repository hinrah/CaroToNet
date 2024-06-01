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

from .train import DataHandler, TRAINING_TRANSFORM, VALIDATION_TRANSFORM

def run_training(data_path, output_path):
    data_handler = DataHandler(data_path, output_path, "all")
    data_handler.initialize()

    utnet = UTNet(1, 32, 3, reduce_size=8, block_list='1234', num_blocks=[1, 1, 1, 1], num_heads=[4, 4, 4, 4], projection='interp', attn_drop=0.1,
                proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
    utnet.cuda()

    lossFunc = BCEWithLogitsLoss()
    opt = Adam(utnet.parameters(), lr=INIT_LR)
    trainer = Trainer(utnet, opt, lossFunc, data_handler.train_loader, None)
    for e in tqdm(range(NUM_EPOCHS)):
        _ = trainer.train_step()
        torch.save(utnet, data_handler.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train 2D vessel segmentation',
                                     description='This trains the 2D vessel segmenation. The training trains 9 models. One with all training data and 8 models '
                                                 'that do not contain the cross-sections at one plane.')
    parser.add_argument('-d', '--data_path')
    parser.add_argument('-o', '--output_path')
    args = parser.parse_args()

    run_training(args.data_path, args.output_path)
