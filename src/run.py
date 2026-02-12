from pipeline.clustering import ClusteringPipeline
from pipeline.download import DownloadPipeline

DownloadPipeline().run()
ClusteringPipeline.run()
