import os.path
from collections import defaultdict

import numpy as np

class SegmentationResults:
    def __init__(self):
        self._results = {}

    def add(self, path, result):
        self._results[path] = result

    @property
    def mean_dice_coefficients(self):
        return [np.mean(list(result.dice_coefficients.values())) for result in self._results.values()]

    @property
    def mean_hausdorff_distance(self):
        return [np.mean(list(result.hausdorff_distances.values())) for result in self._results.values()]

    @property
    def mean_average_contour_distances(self):
        return [np.mean(list(result.average_contour_distances.values())) for result in self._results.values()]

    @property
    def mean_dice_coefficients_of_subjects(self):
        results_by_subject = defaultdict(SegmentationResults)
        for slice_path, result in self._results.items():
            subj_id = os.path.basename(slice_path).split("_")[0]
            results_by_subject[subj_id].add(slice_path, result)
        dice_by_subjects = {}
        for subj_id, results in results_by_subject.items():
            dice_by_subjects[subj_id] = np.mean(results.dice_coefficients(), axis=0)
        return dice_by_subjects

    @property
    def mean_hd_coefficients_of_subjects(self):
        results_by_subject = defaultdict(SegmentationResults)
        for slice_path, result in self._results.items():
            subj_id = os.path.basename(slice_path).split("_")[0]
            results_by_subject[subj_id].add(slice_path, result)
        hd_by_subjects = {}
        for subj_id, results in results_by_subject.items():
            hd_by_subjects[subj_id] = np.mean(results.hausdorff_distances(), axis=0)
        return hd_by_subjects

    @property
    def mean_acd_of_subjects(self):
        results_by_subject = defaultdict(SegmentationResults)
        for slice_path, result in self._results.items():
            subj_id = os.path.basename(slice_path).split("_")[0]
            results_by_subject[subj_id].add(slice_path, result)
        acd_by_subjects = {}
        for subj_id, results in results_by_subject.items():
            acd_by_subjects[subj_id] = np.mean(results.average_contour_distances(), axis=0)
        return acd_by_subjects

    def dice_coefficients(self, class_value=None):
        if class_value:
            return [result.dice_coefficients[class_value] for result in self._results.values()]
        return [list(result.dice_coefficients.values()) for result in self._results.values()]

    def hausdorff_distances(self, class_value=None):
        if class_value:
            return [result.hausdorff_distances[class_value] for result in self._results.values()]
        return [list(result.hausdorff_distances.values()) for result in self._results.values()]

    def average_contour_distances(self, class_value=None):
        if class_value:
            return [result.average_contour_distances[class_value] for result in self._results.values()]
        return np.nan_to_num(np.array([list(result.average_contour_distances.values()) for result in self._results.values()]), nan=np.inf)

    def get_worst_dice_paths(self, class_value):
        dice_coefficients = self.dice_coefficients(class_value=class_value)
        dice_coefficients.sort()

        third_element = dice_coefficients[2]
        paths = []

        for path, result in self._results.items():
            if result.dice_coefficients[class_value] <= third_element:
                paths.append(path)

        return paths

    def get_worst_HD_paths(self, class_value):
        hd = self.hausdorff_distances(class_value=class_value)
        hd.sort()

        third_element = hd[-3]
        paths = []

        for path, result in self._results.items():
            if result.hausdorff_distances[class_value] >= third_element:
                paths.append(path)

        return paths

    def get_worst_ACD_paths(self, class_value):
        acd = self.average_contour_distances(class_value=class_value)
        acd.sort()

        third_element = acd[-3]
        paths = []

        for path, result in self._results.items():
            if result.average_contour_distances[class_value] >= third_element:
                paths.append(path)
        return paths

    def get_acd_by_path(self, path):
        return self._results[path].average_contour_distances

    def get_hd_by_path(self, path):
        return self._results[path].hausdorff_distances

    def get_dice_by_path(self, path):
        return self._results[path].dice_coefficients