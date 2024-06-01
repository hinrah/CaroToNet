import argparse
import os.path

import numpy as np
from glob import glob
from os import path, makedirs
import nibabel as nib

from segmentation_evaluator import SegmentationEvaluator2D
from segmentation_results import SegmentationResults

import matplotlib.pyplot as plt
import pandas as pd


def get_pixel_size(data):
    return data.header['pixdim'][1:3]


class VesselSegmentationEvaluationPlotter:
    def __init__(self, output_folder):
        if not path.exists(output_folder):
            makedirs(output_folder)
        self._output_folder = output_folder

    def _plot(self, filename, y_axis, metric_results):
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        ax.set_xticklabels(['wall', 'lumen'])
        ax.set_ylabel(y_axis)
        plt.subplots_adjust(left=0.215)
        _ = plt.boxplot(np.array(metric_results)[:, 1:], widths=0.5, whis=[1, 99])
        plt.savefig(path.join(self._output_folder, f"{filename}.png"))

    def create_plots(self, segmentation_results):
        self._plot("Dice Coefficients", "DC", segmentation_results.dice_coefficients())
        self._plot("Hausdorff Distance", "HD in mm", segmentation_results.hausdorff_distances())
        self._plot("Average Contour", "ACD in mm", segmentation_results.average_contour_distances())

    @staticmethod
    def show():
        plt.show()


def evaluate_segmentations(prediction_folder, gt_folder, num_slices):
    evaluator = SegmentationEvaluator2D(classes=[0, 1, 2])
    segmentation_results = {}

    keys = list(range(1, num_slices + 1)) + ["all"]

    for i in keys:
        segmentation_results[i] = SegmentationResults()

        if i == "all":
            prediction_paths = glob(path.join(prediction_folder, f"*.nii.gz"))
        else:
            prediction_paths = glob(path.join(prediction_folder, f"*_slice{i}_*.nii.gz"))

        for prediction_path in prediction_paths:
            ground_truth_path = path.join(gt_folder, path.basename(prediction_path))

            gt_data = nib.load(ground_truth_path)
            prediction_data = nib.load(prediction_path)

            segmentation_result = evaluator.evaluate(gt_data.get_fdata().squeeze(), prediction_data.get_fdata().squeeze(), get_pixel_size(gt_data))

            segmentation_results[i].add(prediction_path, segmentation_result)

    return segmentation_results


def save_results_to_xlsx(segmentation_results, path_to_save, num_slices):
    keys = list(range(1, num_slices + 1)) + ["all"]
    metrics_data = {
        "slice": [f"slice{i}" for i in keys],
        "dice_wall": [np.mean(segmentation_results[i].dice_coefficients(), axis=0)[1] for i in keys],
        "dice_lumen": [np.mean(segmentation_results[i].dice_coefficients(), axis=0)[2] for i in keys],
        "hausdorff_distance_wall": [np.mean(segmentation_results[i].hausdorff_distances(), axis=0)[1] for i in keys],
        "hausdorff_distance_lumen": [np.mean(segmentation_results[i].hausdorff_distances(), axis=0)[2] for i in keys],
        "average_countour_distance_wall": [np.mean(segmentation_results[i].average_contour_distances(), axis=0)[1] for i in keys],
        "average_countour_distance_lumen": [np.mean(segmentation_results[i].average_contour_distances(), axis=0)[2] for i in keys],
        "median_dice_wall": [np.median(segmentation_results[i].dice_coefficients(), axis=0)[1] for i in keys],
        "median_dice_lumen": [np.median(segmentation_results[i].dice_coefficients(), axis=0)[2] for i in keys],
        "median_hausdorff_distance_wall": [np.median(segmentation_results[i].hausdorff_distances(), axis=0)[1] for i in keys],
        "median_hausdorff_distance_lumen": [np.median(segmentation_results[i].hausdorff_distances(), axis=0)[2] for i in keys],
        "median_average_countour_distance_wall": [np.median(segmentation_results[i].average_contour_distances(), axis=0)[1] for i in keys],
        "median_average_countour_distance_lumen": [np.median(segmentation_results[i].average_contour_distances(), axis=0)[2] for i in keys],
    }
    df = pd.DataFrame(metrics_data)
    df.to_excel(path_to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Evaluate 2D vessel segmentation',
                                     description='This plots the HD, ACD and DC of the test data and creates an xlsx file with the metrics distributed by slices')
    parser.add_argument('-p', '--prediction_path')
    parser.add_argument('-s', '--save_path')
    parser.add_argument('-gt', '--ground_truth_path')
    parser.add_argument('-n', '--num_slices', type=int)
    args = parser.parse_args()

    segmentation_results = evaluate_segmentations(args.prediction_path, args.ground_truth_path, args.num_slices)

    evaluation_plotter = VesselSegmentationEvaluationPlotter(args.save_path)
    evaluation_plotter.create_plots(segmentation_results["all"])

    save_results_to_xlsx(segmentation_results, os.path.join(args.save_path, "results.xlsx"), args.num_slices)

    worst_paths = set(segmentation_results["all"].get_worst_ACD_paths(1) +
                      segmentation_results["all"].get_worst_ACD_paths(2) +
                      segmentation_results["all"].get_worst_HD_paths(1) +
                      segmentation_results["all"].get_worst_HD_paths(2) +
                      segmentation_results["all"].get_worst_dice_paths(1) +
                      segmentation_results["all"].get_worst_dice_paths(2))

    worst_eval_list = []
    for path in worst_paths:
        worst_eval_list.append({"slice": os.path.basename(path),
                                "hd_wall": segmentation_results["all"].get_hd_by_path(path)[1],
                                "hd_lumen": segmentation_results["all"].get_hd_by_path(path)[2],
                                "acd_wall": segmentation_results["all"].get_acd_by_path(path)[1],
                                "acd_lumen": segmentation_results["all"].get_acd_by_path(path)[2],
                                "dc_wall": segmentation_results["all"].get_dice_by_path(path)[1],
                                "dc_lumen": segmentation_results["all"].get_dice_by_path(path)[2],
                                })
    print(worst_eval_list)


    print([elem["slice"] for elem in worst_eval_list])