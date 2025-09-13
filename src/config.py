from data_models import EvalScriptType

# API Request
EVALSCRIPT_TYPE: EvalScriptType  = "RGB"
GEOMETRY_FILE: str = "blackForestPoly.geojson" # file with geometry to specify AOI
START_DATE: str = "2025-07-01" # date as "YYYY-mm-dd"
END_DATE: str = "2025-07-31" # "now" or YYYY-mm-dd
COLLECTION_ID: str = "sentinel-2-l2a"