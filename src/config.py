from data_sourcing.data_models import CRSType, EvalScriptType

# API Request
EVALSCRIPT_TYPE: EvalScriptType = "INDICES"
GEOMETRY_FILE: str = "aoi_hornisgrinde.geojson"  # file with geometry to specify AOI
GEOMETRY_FILE_CRS: CRSType = "EPSG:4326"
START_DATE: str = "2020-01-01"  # date as "YYYY-mm-dd"
END_DATE: str = "2025-09-30"  # "now" or YYYY-mm-dd
COLLECTION_ID: str = "sentinel-2-l2a"
