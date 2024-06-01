import argparse
from glob import glob
from os import path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import nibabel as nib

from segmentation_evaluator import SegmentationEvaluator2D
from segmentation_results import SegmentationResults


class RoiBinSegmentationEvaluationPlotter:
    def __init__(self, output_folder):
        if not path.exists(output_folder):
            makedirs(output_folder)
        self._output_folder = output_folder

    def _get_mean_metric_by_bins(self, roi_sizes, metric_results):
        (n, bins, patches) = plt.hist(roi_sizes.values(), bins=10, label='hst')

        by_bin = [[] for _ in range(len(bins) - 1)]
        for roi_size, metric in zip(roi_sizes.values(), metric_results):
            for i in range(len(bins) - 1):
                if roi_size >= bins[i] and roi_size < bins[i + 1]:
                    by_bin[i].append(metric)

        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        mean_metric = [np.mean(metrics_per_bin) for metrics_per_bin in by_bin]
        return bin_centers, mean_metric

    def _plot(self, save_name, y_label, roi_sizes, metric_results):
        (n, bins, patches) = plt.hist(roi_sizes.values(), bins=10, label='hst')


        by_bin = [[] for _ in range(len(bins)-1)]
        for roi_size, metric in zip(roi_sizes.values(), metric_results):
            for i in range(len(bins)-1):
                if roi_size >= bins[i] and roi_size < bins[i+1]:
                    by_bin[i].append(metric)

        bin_centers = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        mean_metric = [np.mean(metrics_per_bin) for metrics_per_bin in by_bin]

        bin_centers, mean_lumen_by_bins = self._get_mean_metric_by_bins(roi_sizes, metric_results[:,1])
        _, mean_wall_by_bins = self._get_mean_metric_by_bins(roi_sizes, metric_results[:,2])


        fig1, ax1 = plt.subplots()
        ax1.plot(bin_centers, mean_lumen_by_bins)
        ax1.plot(bin_centers, mean_wall_by_bins)

        fig1.subplots_adjust(left=0.215)
        plt.xlabel("region of interest/patch size")
        plt.ylabel(y_label)
        fig1.savefig(path.join(self._output_folder, f"{save_name}.png"))

    def create_plots(self, roi_sizes, segmentation_results):
        self._plot("Mean Dice Coefficients both", "Mean DC both", roi_sizes, np.array(segmentation_results.dice_coefficients()))
        #self._plot("Mean Dice Coefficients Lumen", "Mean DC Lumen", roi_sizes, segmentation_results.dice_coefficients(1))
        self._plot("Mean Hausdorff Distance both", "Mean HD both in mm", roi_sizes, np.array(segmentation_results.hausdorff_distances()))
        #self._plot("Mean Hausdorff Distance Lumen", "Mean HD Lumen in mm", roi_sizes, segmentation_results.hausdorff_distances(1))
        self._plot("Mean Average Contour both", "Mean ACD both in mm", roi_sizes, np.array(segmentation_results.average_contour_distances()))
        #self._plot("Mean Average Contour Lumen", "Mean ACD Lumen in mm", roi_sizes, segmentation_results.average_contour_distances(1))

    @staticmethod
    def show():
        plt.show()



class RoiSegmentationEvaluationPlotter:
    def __init__(self, output_folder):
        if not path.exists(output_folder):
            makedirs(output_folder)
        self._output_folder = output_folder

    def _plot(self, save_name, y_label, roi_sizes, metric_results, ymin=None, ymax=None):
        fig1, ax1 = plt.subplots()
        by_slice = {
            "slice1": [[], []],
            "slice2": [[], []],
            "slice3": [[], []],
            "slice4": [[], []],
            "slice5": [[], []],
            "slice6": [[], []],
            "slice7": [[], []],
            "slice8": [[], []],
        }
        for filename, roi_size, metric in zip(roi_sizes.keys(), roi_sizes.values(), metric_results):
            for key in by_slice.keys():
                if key in filename:
                    by_slice[key][0].append(roi_size)
                    by_slice[key][1].append(metric)


        roi_sizes_array = np.array(list(roi_sizes.values()))

        stretch = np.max(roi_sizes_array)-np.min(roi_sizes_array)
        sample_points = np.array(range(100))/100*stretch+np.min(roi_sizes_array)

        samples = {sample_point: [] for sample_point in sample_points}

        for roi_size, metric in zip(roi_sizes.values(), metric_results):
            for sample_point in sample_points:
                if np.abs(roi_size-sample_point) < 0.05:
                    samples[sample_point].append(metric)

        running_means = [np.mean(within_distance) for within_distance in samples.values()]

        colors = plt.cm.tab10(range(8))  # ['black', 'grey', 'darkred', 'indianred', 'lightcoral', 'pink', 'mediumorchid', 'orange']
        labels = ["Plane 1", "Plane 2", "Plane 3", "Plane 4", "Plane 5", "Plane 6", "Plane 7", "Plane 8"]

        for idx, (roi_sizes, metrics) in enumerate(by_slice.values()):
            ax1.scatter(roi_sizes, metrics, color=colors[idx], label=labels[idx])

        ax1.plot(sample_points, running_means, linewidth=5, color="black", label="Moving Average")

        if not ymin is None and not ymax is None:
            ax1.set_ylim([ymin, ymax])
        ax1.legend()


        fig1.subplots_adjust(left=0.215)
        plt.xlabel("Region of interest/patch size")
        plt.ylabel(y_label)
        fig1.savefig(path.join(self._output_folder, f"{save_name}.png"))

    def create_plots(self, roi_sizes, segmentation_results):

        self._plot("Dice Coefficients Wall", "DC wall", roi_sizes, segmentation_results.dice_coefficients(1))
        self._plot("Dice Coefficients Lumen", "DC lumen", roi_sizes, segmentation_results.dice_coefficients(2))
        self._plot("Hausdorff Distance Wall", "HD wall in mm", roi_sizes, segmentation_results.hausdorff_distances(1))
        self._plot("Hausdorff Distance Lumen", "HD lumen in mm", roi_sizes, segmentation_results.hausdorff_distances(2))
        self._plot("Average Contour Wall", "ACD wall in mm", roi_sizes, segmentation_results.average_contour_distances(1), ymin=0, ymax=0.9)
        self._plot("Average Contour Lumen", "ACD lumen in mm", roi_sizes, segmentation_results.average_contour_distances(2), ymin=0, ymax=0.9)

    @staticmethod
    def show():
        plt.show()

def get_pixel_size(data):
    return data.header['pixdim'][1:3]

def evaluate_segmentations(prediction_folder, gt_folder):
    evaluator = SegmentationEvaluator2D(classes=[0, 1, 2])

    segmentation_results = SegmentationResults()

    prediction_paths = glob(path.join(prediction_folder, f"*.nii.gz"))

    for prediction_path in prediction_paths:
        ground_truth_path = path.join(gt_folder, path.basename(prediction_path))

        gt_data = nib.load(ground_truth_path)
        prediction_data = nib.load(prediction_path)

        segmentation_result = evaluator.evaluate(gt_data.get_fdata().squeeze(), prediction_data.get_fdata().squeeze(), get_pixel_size(gt_data))

        segmentation_results.add(ground_truth_path, segmentation_result)

    return segmentation_results

def calculate_roi_percentage(patch):
    num_total = patch.size
    num_foreground = np.sum(patch > 0)
    return num_foreground / num_total


def evaluate_roi(gt_folder):
    roi_sizes = {}
    gt_paths = glob(path.join(gt_folder, f"*.nii.gz"))
    for gt_path in gt_paths:
        gt_data = nib.load(gt_path)
        roi_sizes[gt_path] = calculate_roi_percentage(gt_data.get_fdata().squeeze())
    return roi_sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Evaluate segmentation results based on roi size',
                                     description='This plots the HD, ACD and DC of the test data depending on the roi size')
    parser.add_argument('-p', '--prediction_path')
    parser.add_argument('-s', '--save_path')
    parser.add_argument('-gt', '--ground_truth_path')
    args = parser.parse_args()

    plotter = RoiSegmentationEvaluationPlotter(args.save_path)
    #plotter = RoiBinSegmentationEvaluationPlotter(args.save_path)
    segmentation_results = evaluate_segmentations(args.prediction_path, args.ground_truth_path)
    roi_sizes = evaluate_roi(args.ground_truth_path)
    plotter.create_plots(roi_sizes, segmentation_results)

