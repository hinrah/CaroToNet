import math
import os

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

from constants import INIT_LR, NUM_EPOCHS, BATCH_SIZE, SLICES, BASE_OUTPUT, DATA_PATH, IMAGE_KEY, LABEL_KEY, KEYS
from helpers import make_sure_folder_exists, search_for_data, search_for_data_one_channel
from constants import INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, DEVICE
from trainer import Trainer
from model.utnet import UTNet

BASE_OUTPUT = f"/sc-projects/sc-proj-dhzc-icm-carotis/2D_slice_representative/utnet"

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

DUMMY_IN = torch.randn(1, 1, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT).to(DEVICE)
for left_out_slice in ["all"]:
    output_folder = os.path.join(BASE_OUTPUT, f"slice_{left_out_slice}")

    model_path = os.path.join(output_folder, "utnet.pth")
    onnx_path = os.path.join(output_folder, "utnet.onnx")
    make_sure_folder_exists(output_folder)

    X_paths, y_paths = search_for_data(DATA_PATH, left_out_slice=left_out_slice)

    train_images = [paths[0] for paths in X_paths]
    train_masks = [path for path in y_paths]

    train_files = [{IMAGE_KEY: img, LABEL_KEY: mask} for img, mask in zip(train_images, train_masks)]

    train_data_set = CacheDataset(data=train_files, transform=TRAINING_TRANSFORM, cache_rate=1.0,
                                  num_workers=8)

    train_loader = monai.data.DataLoader(train_data_set, shuffle=True, batch_size=BATCH_SIZE)


    net = UTNet(1, 32, 3, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
    net.cuda()

    trainSteps = len(train_data_set) // BATCH_SIZE

    lossFunc = BCEWithLogitsLoss()
    opt = Adam(net.parameters(), lr=INIT_LR)
    trainer = Trainer(net, opt, lossFunc, train_loader, None)

    runs_without_improvement = 0
    for e in tqdm(range(NUM_EPOCHS)):
        totalTrainLoss = trainer.train_step()
        avgTrainLoss = totalTrainLoss / trainSteps
        torch.save(net, model_path)

    net = torch.load(model_path).to(DEVICE)
    torch.onnx.export(net, DUMMY_IN, onnx_path, opset_version=16)
