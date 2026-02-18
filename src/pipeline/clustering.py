import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.preprocessing import MinMaxScaler

import config as cf
from data_processing.feature_service import FeatureService
from data_processing.geometry_processor import GeometryProcessor


class ClusteringPipeline:
    @staticmethod
    def run(n_clusters: int = 4, output_path: str = None) -> np.ndarray:
        gp = GeometryProcessor()
        band_data = gp.flatten_and_filter_monthly_data()

        fs = FeatureService(band_data)
        feature_df = fs.calculate_features_for_monthly_data()

        scaler = MinMaxScaler()
        feature_norm = scaler.fit_transform(feature_df)

        centroids, _ = kmeans_plusplus(
            feature_norm, n_clusters=n_clusters, random_state=20
        )
        kmeans = KMeans(n_clusters=n_clusters, random_state=20, init=centroids)

        labels = kmeans.fit_predict(feature_norm)

        path = (
            output_path
            if output_path is not None
            else "DATA/" + cf.CLUSTER_LABEL_OUTPUT_FILE
        )

        gp.export_reconstruction_as_geotiff(labels, path)

        return labels
