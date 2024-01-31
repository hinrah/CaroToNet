import itertools
import math

import numpy as np
from evalutils.stats import dice_from_confusion_matrix, hausdorff_distance, mean_contour_distance

from metrics import Metrics


class SegmentationEvaluator2D:
    def __init__(self, classes):
        self._classes = classes

    def validate_inputs(self, truth, prediction):
        if not truth.shape == prediction.shape:
            raise ValueError("Ground truth and prediction do not have the same size")
        if not set(truth.flatten()).issubset(self._classes):
            raise ValueError("Truth contains invalid classes")
        if not set(prediction.flatten()).issubset(self._classes):
            raise ValueError("Prediction contains invalid classes")

    def get_confusion_matrix(self, truth, prediction):
        confusion_matrix = np.zeros((len(self._classes), len(self._classes)))
        for class_predicted, class_truth in itertools.product(self._classes, self._classes):
            confusion_matrix[class_truth, class_predicted] = np.sum(
                np.all(np.stack((prediction == class_predicted, truth == class_truth)), axis=0))
        return confusion_matrix

    def evaluate(self, truth: np.ndarray, prediction: np.ndarray, pixel_dim):
        self.validate_inputs(truth, prediction)

        dice_coefficients = self.evaluate_dice_coefficients(truth, prediction)
        hausdorff_distances = self.evaluate_hausdorff_distances(truth, prediction, pixel_dim)
        mean_contour_distances = self.evaluate_mean_contour_distances(truth, prediction, pixel_dim)
        return Metrics(dice_coefficients=dice_coefficients,
                       hausdorff_distances=hausdorff_distances,
                       average_contour_distances=mean_contour_distances)

    def evaluate_dice_coefficients(self, truth, prediction):
        sorted_dice_coefficients = {}
        confusion_matrix = self.get_confusion_matrix(truth, prediction)
        dice_coefficients = dice_from_confusion_matrix(confusion_matrix)
        for i, class_value in enumerate(self._classes):
            sorted_dice_coefficients[class_value] = dice_coefficients[i]
        return sorted_dice_coefficients

    def evaluate_hausdorff_distances(self, truth: np.ndarray, prediction: np.ndarray, pixel_dim):
        hausdorff_distances = {}
        for class_value in self._classes:
            try:
                hausdorff_distances[class_value] = hausdorff_distance(truth == class_value,
                                                                      prediction == class_value,
                                                                      pixel_dim)
            except ValueError:
                hausdorff_distances[class_value] = math.inf
        return hausdorff_distances

    def evaluate_mean_contour_distances(self, truth: np.ndarray, prediction: np.ndarray, pixel_dim):
        mean_contour_distances = {}
        for class_value in self._classes:
            try:
                mean_contour_distances[class_value] = mean_contour_distance(truth == class_value,
                                                                            prediction == class_value,
                                                                            pixel_dim)
            except ValueError:
                mean_contour_distances[class_value] = math.inf
        return mean_contour_distances
