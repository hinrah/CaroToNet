import numpy as np


class SegmentationResults:
    def __init__(self):
        self._results = []

    def add(self, result):
        self._results.append(result)

    @property
    def mean_dice_coefficients(self):
        return [np.mean(list(result.dice_coefficients.values())) for result in self._results]

    @property
    def mean_hausdorff_distance(self):
        return [np.mean(list(result.hausdorff_distances.values())) for result in self._results]

    @property
    def mean_average_contour_distances(self):
        return [np.mean(list(result.average_contour_distances.values())) for result in self._results]

    def dice_coefficients(self, class_value=None):
        if class_value:
            return [result.dice_coefficients[class_value] for result in self._results]
        return [list(result.dice_coefficients.values()) for result in self._results]

    def hausdorff_distances(self, class_value=None):
        if class_value:
            return [result.hausdorff_distances[class_value] for result in self._results]
        return [list(result.hausdorff_distances.values()) for result in self._results]

    def average_contour_distances(self, class_value=None):
        if class_value:
            return [result.average_contour_distances[class_value] for result in self._results]
        return np.nan_to_num(np.array([list(result.average_contour_distances.values()) for result in self._results]), nan=np.inf)
