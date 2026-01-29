from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BandDTO:
    pixel_list: np.ndarray
    spatial_data: np.ndarray
    pixel_coords: np.ndarray
