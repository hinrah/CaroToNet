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

from train import TRAINING_TRANSFORM, VALIDATION_TRANSFORM

set_determinism(1337)
torch.manual_seed(1337)

def build_unet(args):
    unet = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=args.num_channels,
        out_channels=3,
        channels=tuple([int(args.num_filter*math.pow(2,a)) for a in range(args.model_depth)]),
        strides=tuple([2]*args.model_depth-1),
        num_res_units=3,
        dropout=args.dropout,
        norm=Norm.BATCH
    ).to(DEVICE)

    return unet

def determine_data_augmentation(use_augmentation):
    if use_augmentation:
        return TRAINING_TRANSFORM
    else:
        return VALIDATION_TRANSFORM

def train(args, train_loader, validation_loader):
    output_folder = os.path.join(args.base_output, f"dropout{args.dropout}_numFilter{args.num_filter}_modelDepth{args.model_depth}_augmentation{args.augmentation}_fold{args.fold}")

    model_path = os.path.join(output_folder, f"unet.pth")
    onnx_path = os.path.join(output_folder, "unet.onnx")
    make_sure_folder_exists(output_folder)

    unet = build_unet(args)

    trainSteps = len(train_loader.dataset) // BATCH_SIZE
    validationSteps = len(validation_loader.dataset) // BATCH_SIZE
    best_validation_lost = math.inf
    H = {"train_loss": [], "validation_loss": []}

    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=INIT_LR)
    trainer = Trainer(unet, opt, lossFunc, train_loader, validation_loader)

    runs_without_improvement = 0
    for e in tqdm(range(NUM_EPOCHS)):

        totalTrainLoss = trainer.train_step()
        totalValidationLoss = trainer.evaluate_step()

        avgTrainLoss = totalTrainLoss / trainSteps
        avgValidationLoss = totalValidationLoss / validationSteps

        if best_validation_lost > avgValidationLoss:
            best_validation_lost = avgValidationLoss
            torch.save(unet, model_path)
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

    unet = torch.load(model_path)

    return unet

class Config():
    def __init__(self) -> None:
        self.dropout = None
        self.num_filter = None
        self.model_depth = None
        self.augmentation = None
        self.fold = None
        self.base_output = None
        self.num_channels = None


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Hyperparameter evaluation Unet',
                                     description='Perform Grid search for hyperparameter using 5-fold cross-validation')
    parser.add_argument('-d', '--data_path')
    parser.add_argument('-c', '--num_channels', type=int)
    parser.add_argument('-s', '--image_size', type=int)
    parser.add_argument('-o', '--base_output')
    args = parser.parse_args()


    for dropout in [0.1, 0.2, 0.3]:
        for num_filter in [8,16]:
            for model_depth in [5,6]:
                for augmentation in [False, True]:
                    
                    results_path = os.path.join(args.base_output, "results.csv")
                    dice_metric = DiceMetric(include_background=False)
                    hd_metric = HausdorffDistanceMetric(include_background = False)
                    acd_metric = SurfaceDistanceMetric(include_background = False, symmetric=True)


                    for fold in range(5):
                        config = Config()
                        config.dropout = dropout
                        config.num_filter = num_filter
                        config.model_depth = model_depth
                        config.augmentation = augmentation
                        config.fold = fold
                        config.base_output = args.base_output
                        config.num_channels = args.num_channels
                        config.image_size = args.image_size
                        train_loader, validation_loader = load_data(args.data_path, fold, augmentation, args.num_channels==2)
                        unet = train(config, train_loader, validation_loader)

                        mean_validation_metric(dice_metric, unet, validation_loader)
                        mean_validation_metric(hd_metric, unet, validation_loader)
                        mean_validation_metric(acd_metric, unet, validation_loader)

                    mean_dices = torch.nanmean(dice_metric.get_buffer(), axis=0).cpu().numpy()
                    mean_acd = torch.nanmean(torch.where(torch.isinf(acd_metric.get_buffer()), torch.nan, acd_metric.get_buffer()),
                                             axis=0).cpu().numpy() * 0.195
                    mean_hd = torch.nanmean(torch.where(torch.isinf(hd_metric.get_buffer()), torch.nan, hd_metric.get_buffer()), axis=0).cpu().numpy() * 0.195
                    mean_hd_inf = torch.mean(hd_metric.get_buffer(), axis=1)
                    missing_slices = int(torch.sum(torch.isinf(mean_hd_inf)).cpu().numpy())

                    with open(results_path, 'a', newline="") as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(
                            [dropout, num_filter, model_depth, augmentation, mean_dices[0], mean_dices[1], mean_hd[0], mean_hd[1], mean_acd[0], mean_acd[1],
                             missing_slices])
                        print([dropout, num_filter, model_depth, augmentation, mean_dices[0], mean_dices[1], mean_hd[0], mean_hd[1], mean_acd[0], mean_acd[1],
                               missing_slices])
                    dice_metric.reset()
                    hd_metric.reset()
                    acd_metric.reset()