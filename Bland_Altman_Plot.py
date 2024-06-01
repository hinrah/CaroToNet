import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
from glob import glob
import os
import nibabel as nib
from evalutils.stats import hausdorff_distance


def create_wall_dist_dict(folder):
    files = list(glob(os.path.join(folder, "*.nii.gz")))
    files.sort()
    distances = []

    for file in files:
        label = nib.load(file).get_fdata().squeeze()
        distance = hausdorff_distance(label == 1, label == 2, nib.load(file).header["pixdim"][1:3])
        distances.append(distance)

    return {
        "file_name": [os.path.basename(file) for file in files],
        "max_wall_thickness": distances
    }

distances_gt = create_wall_dist_dict("C:\\Users\\hinrich\\Desktop\\2D_slice_representative\\2D_Data\\Dataset019_2dT1Centerlinedist25mmRealCentered\\labelsTs")
distances_model = create_wall_dist_dict("C:\\Users\\hinrich\\Desktop\\2D_slice_representative\\JMI_results_v3\\predictions\\unet\\original")

all_gt = np.array(distances_gt["max_wall_thickness"])
all_model = np.array(distances_model["max_wall_thickness"])

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
for filename, gt, model in zip(distances_gt["file_name"], distances_gt["max_wall_thickness"], distances_model["max_wall_thickness"]):
    for key in by_slice.keys():
        if key in filename:
            by_slice[key][0].append(gt)
            by_slice[key][1].append(model)

colors = plt.cm.tab10(range(8))  # ['black', 'grey', 'darkred', 'indianred', 'lightcoral', 'pink', 'mediumorchid', 'orange']
labels = ["Plane 1", "Plane 2", "Plane 3", "Plane 4", "Plane 5", "Plane 6", "Plane 7", "Plane 8"]

for idx, distances in enumerate(by_slice.values()):
    distances = np.array(distances)
    mean_value_of_two_ratings = np.mean(distances, axis=0)
    difference_between_two_ratings = distances[0] - distances[1]
    plt.scatter(mean_value_of_two_ratings, difference_between_two_ratings, color=colors[idx], label=labels[idx])
plt.legend()

difference_between_two_ratings = all_gt - all_model

std = np.std(difference_between_two_ratings)
mean = np.mean(difference_between_two_ratings)
print(f"mean: {mean}")

num_slices = len(all_gt)
exam = list(range(num_slices)) * 2
judge = ["A"] * num_slices + ["B"] * num_slices
rating = list(all_gt) + list(all_model)

df = pd.DataFrame({'exam': exam,
                   'judge': judge,
                   'rating': rating})
icc = pg.intraclass_corr(data=df, targets='exam', raters='judge', ratings='rating')

icc.set_index('Type')
print(icc)
print(mean)
plt.axhline(mean, color='gray', linestyle='--', lw=0.4)
plt.axhline(mean + 1.96 * std, color='gray', linestyle='--', lw=0.4)
plt.axhline(mean - 1.96 * std, color='gray', linestyle='--', lw=0.4)
plt.xlabel("Mean $max(VWT)$ [mm]")
plt.ylabel(r'Ground truth $max(VWT)$ - $M_R$ $max(VWT)$ [mm]')
plt.show()

if __name__ == "__main__":
    pass
