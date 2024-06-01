import os
from glob import glob


def make_sure_folder_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


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

def search_for_data_one_channel(base_dir):
    print(f"{base_dir}/imagesTr/*_0000.nii.gz")
    X_paths = sorted(list(glob(f"{base_dir}/imagesTr/*_0000.nii.gz")))

    y_paths = sorted(list(glob(f"{base_dir}/labelsTr/*.nii.gz")))
    return X_paths, y_paths