import math
import os
import csv

from torch.nn import BCEWithLogitsLoss
import torch
from torch.optim import Adam
from tqdm import tqdm
import monai.transforms as mt
from monai.data import CacheDataset
import monai.data
from monai.networks.layers.factories import Norm
from monai.utils.misc import set_determinism
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
import numpy as np
import argparse

from constants import INIT_LR, NUM_EPOCHS, BATCH_SIZE, IMAGE_KEY, LABEL_KEY, KEYS
from helpers import make_sure_folder_exists, search_for_data, search_for_data_one_channel
from constants import DEVICE
from trainer import Trainer
from model.utnet import UTNet
from train import TRAINING_TRANSFORM, VALIDATION_TRANSFORM

set_determinism(1337)
torch.manual_seed(1337)

def determine_data_augmentation(use_augmentation):
    if use_augmentation:
        return TRAINING_TRANSFORM
    else:
        return VALIDATION_TRANSFORM

def train(net, args, train_loader, validation_loader):
    output_folder = os.path.join(args.base_output, f"utnet_{args.name}_fold{args.fold}")

    model_path = os.path.join(output_folder, f"utnet.pth")
    make_sure_folder_exists(output_folder)

    trainSteps = len(train_loader.dataset) // BATCH_SIZE
    validationSteps = len(validation_loader.dataset) // BATCH_SIZE
    best_validation_lost = math.inf
    H = {"train_loss": [], "validation_loss": []}

    lossFunc = BCEWithLogitsLoss()
    opt = Adam(net.parameters(), lr=INIT_LR)
    trainer = Trainer(net, opt, lossFunc, train_loader, validation_loader)

    runs_without_improvement = 0
    for e in tqdm(range(NUM_EPOCHS)):

        totalTrainLoss = trainer.train_step()
        totalValidationLoss = trainer.evaluate_step()

        avgTrainLoss = totalTrainLoss / trainSteps
        avgValidationLoss = totalValidationLoss / validationSteps

        if best_validation_lost > avgValidationLoss:
            best_validation_lost = avgValidationLoss
            torch.save(net, model_path)
            print("saved")
            runs_without_improvement = 0
        else:
            runs_without_improvement += 1
        if runs_without_improvement > 50:
            break

        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["validation_loss"].append(avgValidationLoss.cpu().detach().numpy())
        print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
        print("Train loss: {}, validation loss: {}".format(
            avgTrainLoss, avgValidationLoss))

    net = torch.load(model_path)
    return net

class Config():
    def __init__(self) -> None:
        self.dropout = None
        self.num_filter = None
        self.model_depth = None
        self.augmentation = None
        self.fold = None
        self.base_output = None
        self.num_channels = None
        self.name = None


def load_data(data_path, fold_id, use_augmentation, use_distance):
    if use_distance:
        X_paths, y_paths = search_for_data(data_path, left_out_slice="None")
    else:
        X_paths, y_paths = search_for_data_one_channel(data_path)
    
    pat_ids = list(set([os.path.basename(path).split("_")[0] for path in y_paths]))
    pat_ids.sort()
    pat_ids = np.array(pat_ids)
    folds = list(np.array_split(pat_ids, 5))
    val_ids = folds.pop(fold_id)
    train_ids = [pat_id for fold in folds for pat_id in fold]

    if use_distance:
        train_images = [paths for paths in X_paths if os.path.basename(paths[0]).split("_")[0] in train_ids]
        validation_images = [paths for paths in X_paths if os.path.basename(paths[0]).split("_")[0] in val_ids]
    else:
        train_images = [paths for paths in X_paths if os.path.basename(paths).split("_")[0] in train_ids]
        validation_images = [paths for paths in X_paths if os.path.basename(paths).split("_")[0] in val_ids]

    train_masks = [path for path in y_paths if os.path.basename(path).split("_")[0] in train_ids]
    validation_masks = [path for path in y_paths if os.path.basename(path).split("_")[0] in val_ids]
    train_files = [{IMAGE_KEY: img, LABEL_KEY: mask} for img, mask in zip(train_images, train_masks)]
    val_files = [{IMAGE_KEY: img, LABEL_KEY: mask} for img, mask in zip(validation_images, validation_masks)]

    train_data_set = CacheDataset(data=train_files, transform=determine_data_augmentation(use_augmentation), cache_rate=1.0,
                                  num_workers=8)
    validation_data_set = CacheDataset(data=val_files, transform=VALIDATION_TRANSFORM, cache_rate=1.0,
                                       num_workers=8)

    train_loader = monai.data.DataLoader(train_data_set, shuffle=True, batch_size=BATCH_SIZE)
    validation_loader = monai.data.DataLoader(validation_data_set, shuffle=False, batch_size=BATCH_SIZE)
    return train_loader, validation_loader


def mean_validation_metric(metric, network, validation_loader):
    with torch.no_grad():
        network.eval()
        for batch_data in validation_loader:
            x, y = batch_data[IMAGE_KEY], batch_data[LABEL_KEY]
            prediction = network(x)
            binary_prediction = torch.nn.functional.one_hot(prediction.argmax(1), num_classes=3).transpose(3,1).transpose(3,2)

            metric(y_pred = binary_prediction, y=y)


def build_simple_utnet():
    return UTNet(1, 32, 3, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)

def build_middle_utnet():
    return UTNet(1, 32, 3, reduce_size=8, block_list='1234', num_blocks=[1,1,4,8], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)

def build_complex_utnet():
    return UTNet(1, 32, 3, reduce_size=8, block_list='234', num_blocks=[2,4,8], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Hyperparameter evaluation Utnet',
                                     description='Perform Grid search for hyperparameter using 5-fold cross-validation')
    parser.add_argument('-d', '--data_path')
    parser.add_argument('-c', '--num_channels', type=int)
    parser.add_argument('-s', '--image_size', type=int)
    parser.add_argument('-o', '--base_output')
    args = parser.parse_args()

    networks = [("simple", build_simple_utnet),
                ("middle", build_middle_utnet),
                ("complex", build_complex_utnet),
    ]
    for name, net_builder in networks:                    
        results_path = os.path.join(args.base_output, "results.csv")
        dice_metric = DiceMetric(include_background=False)
        hd_metric = HausdorffDistanceMetric(include_background = False)
        acd_metric = SurfaceDistanceMetric(include_background = False, symmetric=True)

        for fold in range(5):
            config = Config()
            config.fold = fold
            config.base_output = args.base_output
            config.num_channels = args.num_channels
            config.image_size = args.image_size
            config.name = name
            train_loader, validation_loader = load_data(args.data_path, fold, True, args.num_channels==2)
            net = train(net_builder().cuda(), config, train_loader, validation_loader)

            mean_validation_metric(dice_metric, net, validation_loader)
            mean_validation_metric(hd_metric, net, validation_loader)
            mean_validation_metric(acd_metric, net, validation_loader)

        mean_dices = torch.nanmean(dice_metric.get_buffer(), axis=0).cpu().numpy()
        mean_acd = torch.nanmean(torch.where(torch.isinf(acd_metric.get_buffer()), torch.nan, acd_metric.get_buffer()),
                                    axis=0).cpu().numpy() * 0.195
        mean_hd = torch.nanmean(torch.where(torch.isinf(hd_metric.get_buffer()), torch.nan, hd_metric.get_buffer()), axis=0).cpu().numpy() * 0.195
        mean_hd_inf = torch.mean(hd_metric.get_buffer(), axis=1)
        missing_slices = int(torch.sum(torch.isinf(mean_hd_inf)).cpu().numpy())

        with open(results_path, 'a', newline="") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(
                [name, mean_dices[0], mean_dices[1], mean_hd[0], mean_hd[1], mean_acd[0], mean_acd[1],
                    missing_slices])
            print([name, mean_dices[0], mean_dices[1], mean_hd[0], mean_hd[1], mean_acd[0], mean_acd[1],
                    missing_slices])
        dice_metric.reset()
        hd_metric.reset()
        acd_metric.reset()