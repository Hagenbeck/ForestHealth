from data_sourcing.data_models import CRSType, EvalScriptType

# API Request
EVALSCRIPT_TYPE: EvalScriptType = "INDICES"
GEOMETRY_FILE: str = "aoi_hornisgrinde.geojson"  # file with geometry to specify AOI
GEOMETRY_FILE_CRS: CRSType = "EPSG:4326"
WORLDCOVER_FILE: str = "ESA_WorldCover_10m_2021_v200_N48E006_Map.tif"
WORLDCOVER_FILE_RESOLUTION: int = 10
DEM_FILE: str = "ASTGTMV003_N48E008_dem.tif"
DEM_FILE_RESOLUTION: int = 30
OBSERVATION_SAVE_FILE: str = "dataMonthly.npy"
CLUSTER_LABEL_OUTPUT_FILE: str = "cluster_labels.tif"
START_DATE: str = "2020-01-01"  # date as "YYYY-mm-dd"
END_DATE: str = "2025-09-30"  # "now" or YYYY-mm-dd
COLLECTION_ID: str = "sentinel-2-l2a"
RESOLUTION: int = 20
