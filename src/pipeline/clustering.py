import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.preprocessing import MinMaxScaler

import config as cf
from core.logger import Logger, LogSegment
from data_processing.feature_service import FeatureService
from data_processing.geometry_processor import GeometryProcessor


class ClusteringPipeline:
    @staticmethod
    def run(n_clusters: int = 4, output_path: str = None) -> np.ndarray:
        logger = Logger.get_instance()
        logger.info(LogSegment.CLUSTERING, f"Starting clustering pipeline with {n_clusters} clusters")
        
        gp = GeometryProcessor()
        logger.info(LogSegment.CLUSTERING, "Flattening and filtering monthly data")
        band_data = gp.flatten_and_filter_monthly_data()

        fs = FeatureService(band_data)
        logger.info(LogSegment.CLUSTERING, "Calculating features for clustering")
        feature_df = fs.calculate_features_for_monthly_data()

        logger.info(LogSegment.CLUSTERING, "Normalizing features with MinMaxScaler")
        scaler = MinMaxScaler()
        feature_norm = scaler.fit_transform(feature_df)

        logger.info(LogSegment.CLUSTERING, f"Initializing K-means++ with {n_clusters} clusters")
        centroids, _ = kmeans_plusplus(
            feature_norm, n_clusters=n_clusters, random_state=20
        )
        kmeans = KMeans(n_clusters=n_clusters, random_state=20, init=centroids)

        logger.info(LogSegment.CLUSTERING, "Running K-means clustering")
        labels = kmeans.fit_predict(feature_norm)

        path = (
            output_path
            if output_path is not None
            else "DATA/" + cf.CLUSTER_LABEL_OUTPUT_FILE
        )

        logger.info(LogSegment.CLUSTERING, f"Exporting cluster labels to {path}")
        gp.export_reconstruction_as_geotiff(labels, path)
        logger.info(LogSegment.CLUSTERING, "Clustering pipeline completed successfully")

        return labels
