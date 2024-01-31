import dataclasses


@dataclasses.dataclass
class Metrics:
    dice_coefficients: dict
    hausdorff_distances: dict
    average_contour_distances: dict
