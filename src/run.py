import numpy as np

from pipeline.download import DownloadPipeline

data = DownloadPipeline().run()

np.save("download.npy", data)
