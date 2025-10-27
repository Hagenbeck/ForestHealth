import numpy as np

from pipeline.download import download_sentinel_data
from pipeline.train_classifier import train_forest_classificator

data = download_sentinel_data()

np.save("download.npy", data)
