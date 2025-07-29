from pathlib import Path
import numpy as np
import sys
from src.pipeline.download import download_sentinel_data

sys.path.append(str(Path.cwd().parents[1]))

data  = download_sentinel_data()

np.save("data.npy", data)