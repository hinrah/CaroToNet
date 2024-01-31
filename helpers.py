import os
from glob import glob
from pathlib import Path


def make_sure_folder_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def paths_exist(paths):
    for path in paths:
        if not os.path.exists(path):
            return False
    return True


def search_for_data(base_dir, left_out_slice):
    print(f"{base_dir}/imagesTr/*_0000.nii.gz")
    X_paths = sorted(list(glob(f"{base_dir}/imagesTr/*_0000.nii.gz")))
    X_paths = [path for path in X_paths if f"_slice{left_out_slice}_" not in path]
    X_paths = [[t1_path, t1_path.replace("_0000", "_0001")] for t1_path in X_paths]

    y_paths = sorted(list(glob(f"{base_dir}/labelsTr/*.nii.gz")))
    y_paths = [path for path in y_paths if f"_slice{left_out_slice}_" not in path]

    return X_paths, y_paths
