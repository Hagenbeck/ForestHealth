from typing import Literal

# API Request
EVALSCRIPT_TYPE: Literal["RGB", "ALL", "INDICES"] = "RGB"
GEOMETRY_FILE: str = "blackForestPoly.geojson" # file with geometry to specify AOI
START_DATE: str = "2020-01-01" # date as "YYYY-mm-dd"
END_DATE: str = "now" # "now" or YYYY-mm-dd