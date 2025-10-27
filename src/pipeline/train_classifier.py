import numpy as np
import pandas as pd

from core.geometry import get_slicing_for_subarray, load_mask_from_geojson
from core.paths import get_data_path


def train_forest_classificator(
    paths_to_data_file: str = "data2025.npy",
    path_to_ground_truth_labels: str = "forest_labels.geojson",
    path_to_AOI_bounds: str = "aoi_hornisgrinde.geojson",
):
    """_summary_

    Args:
        paths_to_data_file (str, optional): _description_. Defaults to "data2025.npy".
        path_to_ground_truth_labels (str, optional): _description_. Defaults to "forest_labels.geojson".
        path_to_AOI_bounds (str, optional): _description_. Defaults to "aoi_hornisgrinde.geojson".
    """
    training_data = _retrieve_training_AOI(
        paths_to_data_file=paths_to_data_file,
        path_to_ground_truth_labels=path_to_ground_truth_labels,
        path_to_AOI_bounds=path_to_AOI_bounds,
    )


def predict_forest_areas(): ...


def _retrieve_training_AOI(
    paths_to_data_file: str,
    path_to_ground_truth_labels: str,
    path_to_AOI_bounds: str,
) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        paths_to_data_file (str): _description_
        path_to_ground_truth_labels (str): _description_
        path_to_AOI_bounds (str): _description_

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    data_path_to_ground_truth_labels = get_data_path(
        filename=path_to_ground_truth_labels
    )
    forest_mask, label_transform = load_mask_from_geojson(
        path=data_path_to_ground_truth_labels
    )

    aoi_array = np.load(get_data_path(paths_to_data_file), mmap_mode="r")

    min_col, max_col, min_row, max_row = get_slicing_for_subarray(
        path_to_labels=path_to_ground_truth_labels, path_to_aoi=path_to_AOI_bounds
    )

    sliced_aoi_array = aoi_array[:, :, min_row:max_row, min_col:max_col]

    train_slice = np.squeeze(sliced_aoi_array)
    train_slice = np.transpose(train_slice, (1, 2, 0))

    forest_mask = forest_mask[:, :, np.newaxis]
    combined_training_data = np.concatenate([train_slice, forest_mask], axis=2)

    return combined_training_data


def _transform_spatial_data_to_dataframe(spatial_data: np.ndarray) -> pd.DataFrame:
    """_summary_

    Args:
        spatial_data (np.ndarray): _description_

    Returns:
        pd.DataFrame: _description_
    """
    ...


def _prepare_data_for_classification(): ...
