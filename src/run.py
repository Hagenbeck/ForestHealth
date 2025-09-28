import numpy as np

from pipeline.download import download_sentinel_data

data = download_sentinel_data()

np.save("data2020.npy", data)
