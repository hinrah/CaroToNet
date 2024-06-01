import itertools
import math

import numpy as np
from evalutils.stats import dice_from_confusion_matrix, hausdorff_distance, mean_contour_distance

from metrics import Metrics


class RoiSizeEvaluator:


    def evaluate_roi_percentage(self, patch):
        num_total = patch.size
        num_foreground = np.sum(patch>0)
        return num_foreground/num_total