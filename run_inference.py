import argparse
import os
import torch

from monai.data import Dataset
from glob import glob
import monai.transforms as mt
import numpy as np
import nibabel as nib

from helpers import make_sure_folder_exists

IMAGE_KEY = "image"

DEVICE = "cpu"

test_transform = mt.Compose(
    [
        mt.LoadImageD(IMAGE_KEY, reader="monai.data.NibabelReader", ensure_channel_first=True, image_only=False),
        mt.SqueezeDimd(IMAGE_KEY, dim=-1),
        mt.SqueezeDimd(IMAGE_KEY, dim=-1),
        mt.SqueezeDimd(IMAGE_KEY, dim=-1),
        mt.ToDeviced(IMAGE_KEY, DEVICE),
        mt.NormalizeIntensityd(IMAGE_KEY, channel_wise=True)
    ]
)


def search_for_test_data(image_folder, slice_modifier):
    x_paths = sorted(list(glob(f"{image_folder}/{slice_modifier}_0000.nii.gz")))
    x_paths = [[t1_path, t1_path.replace("_0000", "_0001")] for t1_path in x_paths]
    return x_paths


def save_as_nifti(prediction, image_meta, prediction_folder):
    prediction_mask = np.int16(np.argmax(prediction.cpu().detach()[0], axis=0))

    prediction_path = os.path.join(prediction_folder, os.path.basename(image_meta["filename_or_obj"]).replace("_0000.nii", ".nii"))
    world_matrix = image_meta["affine"]

    img = nib.Nifti1Image(prediction_mask.reshape((128, 128, 1, 1, 1, 1)), world_matrix)
    nib.save(img, prediction_path)


def run_inference(model_path, eval_images, prediction_folder):
    make_sure_folder_exists(prediction_folder)
    unet = torch.load(model_path, map_location=torch.device(DEVICE))
    _ = unet.eval()

    eval_files = [{IMAGE_KEY: img} for img in eval_images]
    eval_data_set = Dataset(data=eval_files, transform=test_transform)

    for eval_data in eval_data_set:

        prediction = unet(eval_data[IMAGE_KEY].reshape((1, 1, 128, 128)))
        save_as_nifti(prediction, eval_data["image_meta_dict"], prediction_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run Inference',
                                     description='This runs the inference on a dataset and saves it as .nii.gz')
    parser.add_argument('-i', '--image_folder')
    parser.add_argument('-m', '--model_path')
    parser.add_argument('-o', '--output_folder')
    parser.add_argument('-s', '--slice_modifier', default="[1-8]", required=False)
    args = parser.parse_args()

    image_files = search_for_test_data(args.image_folder, args.slice_modifier)

    run_inference(args.model_path, image_files, args.output_folder)
