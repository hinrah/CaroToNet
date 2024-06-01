import argparse
import os.path
from collections import defaultdict

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


class SubjectSegmentationEvaluationPlotter:
    def __init__(self, output_folder):
        if not path.exists(output_folder):
            makedirs(output_folder)
        self._output_folder = output_folder

    def _plot(self, filename, y_axis, metric_results):
        import seaborn
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        ax.set_ylabel(y_axis)
        ax.set_xticklabels(['wall', 'lumen'])
        #seaborn.violinplot(np.array(metric_results)[:, 1:])
        a = plt.boxplot(np.array(metric_results)[:, 1:], widths=0.5, whis=[5, 95])
        plt.subplots_adjust(left=0.245)
        plt.savefig(path.join(self._output_folder, f"{filename}.png"))

    def create_plots(self, segmentation_results):
        self._plot("Dice Coefficients", "Mean DC of subject", np.array(list(segmentation_results.mean_dice_coefficients_of_subjects.values())))
        self._plot("Hausdorff Distance", "Mean HD of subject in mm", np.array(list(segmentation_results.mean_hd_coefficients_of_subjects.values())))
        self._plot("Average Contour", "Mean ACD of subject in mm", np.array(list(segmentation_results.mean_acd_of_subjects.values())))

    @staticmethod
    def show():
        plt.show()


def evaluate_segmentations(prediction_folder, gt_folder):
    evaluator = SegmentationEvaluator2D(classes=[0, 1, 2])
    segmentation_results = SegmentationResults()
    prediction_paths = glob(path.join(prediction_folder, f"*.nii.gz"))
    for prediction_path in prediction_paths:
        ground_truth_path = path.join(gt_folder, path.basename(prediction_path))

        gt_data = nib.load(ground_truth_path)
        prediction_data = nib.load(prediction_path)

        segmentation_result = evaluator.evaluate(gt_data.get_fdata().squeeze(), prediction_data.get_fdata().squeeze(), get_pixel_size(gt_data))

        segmentation_results.add(prediction_path, segmentation_result)

    return segmentation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Evaluate 2D vessel segmentation',
                                     description='This plots the HD, ACD and DC of the test data and creates an xlsx file with the metrics distributed by slices')
    parser.add_argument('-p', '--prediction_path')
    parser.add_argument('-s', '--save_path')
    parser.add_argument('-gt', '--ground_truth_path')
    args = parser.parse_args()

    segmentation_results = evaluate_segmentations(args.prediction_path, args.ground_truth_path)

    evaluation_plotter = SubjectSegmentationEvaluationPlotter(args.save_path)
    evaluation_plotter.create_plots(segmentation_results)
