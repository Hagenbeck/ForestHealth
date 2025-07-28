import src.config as conf
import numpy as np

from datetime import datetime
from src.utils.date_helper import parse_date
from src.utils.evalscripts import get_evalscript, get_response_setup
from src.data_models import EvalScriptType
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer

def build_json_request(width_px: int, 
                       height_px: int, 
                       start_date: datetime, 
                       end_date: datetime, 
                       evalscript_type: EvalScriptType = "RGB", 
                       bbox: list[float] | None = None, 
                       geometry: dict | None = None) -> dict:
    """
    Builds the JSON Request for a request to the sentinelhub API.

    Parameters
    ----------
    width_px: int
        Width of result in pixels. Maximum 2500
    height_px: int
        Height of result in pixels. Maximum 2500
    start_date: datetime
        Start date for request
    end_date: datetime
        End date for request
    evalscript_type: EvalScriptType
        Specifies use case of request. Either "RGB", "ALL" or "INDICES"
    bbox: list[float] | None
        Specifiy bounds of request. Gets preffered over geometry
    geometry: dict | None
        Specifiy bounds of request

    Returns
    -------
    dict
        Request for the specified parameters in JSON format.
    """
    
    if width_px > 2500:
        raise ValueError(f"The API allows for a maximum of 2500 pixels. {width_px} is too wide.")
    elif height_px > 2500:
        raise ValueError(f"The API allows for a maximum of 2500 pixels. {width_px} is too high.")
    
    evalscript = get_evalscript(evalscript_type)
    responses = get_response_setup(evalscript_type)
    
    if evalscript_type == "INDICES":
        processing_block = { "mosaicking": "ORBIT" }
        data_filter = {
            'timeRange': {
                'from': f'{start_date.strftime("%Y-%m-%d")}T00:00:00Z',
                'to': f'{end_date.strftime("%Y-%m-%d")}T23:59:59Z'
            }
        }
    else:
        processing_block = {}  # Default to SIMPLE
        data_filter = {
            'timeRange': {
                'from': f'{start_date.strftime("%Y-%m-%d")}T00:00:00Z',
                'to': f'{end_date.strftime("%Y-%m-%d")}T23:59:59Z'
            },
            'mosaickingOrder': 'leastCC',
            'maxCloudCoverage': 20  # Optional but helpful
        }
    
    json_request = {
                    'input': {
                        'bounds': {
                            'properties': {
                                'crs': 'http://www.opengis.net/def/crs/OGC/1.3/CRS84'
                            }
                        },
                        'data': [
                                    {
                                        'type': conf.COLLECTION_ID.upper(),
                                        'dataFilter': data_filter,
                                        'processing': processing_block
                                    }
                                ]
                    },
                    'output': {
                        'width': width_px,
                        'height': height_px,
                        'responses': responses
                    },
                    'evalscript': evalscript
                }
    
    if bbox is None and geometry is None:
        raise ValueError("Either 'bbox' or 'geometry' must be provided.")
    elif bbox is not None:
        json_request["input"]["bounds"]["bbox"] = bbox
    else:
        json_request["input"]["bounds"]["geometry"] = geometry
    
    
    return json_request

def get_tiling_bounds(geometry: dict, resolution: int = 20, dimension: int = 2500) -> np.ndarray:
    """
    Calculates the tiles needed to fetch data from the sentinelhub API at the highest resolution.

    Parameters
    ----------
    geometry : dict
        Dictionary out of a geojson that represents the AOI
    resolution : int
        The resolution that each pixel should have. Standard is 20
    dimension: int
        The maximum dimension that is allowed in a request. Standard is 2500

    Returns
    -------
    np.ndarray
        Array with the corners of all tiles
    """
    
    geom = shape(geometry)
    project = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    geom_m = transform(project, geom)

    minx, miny, maxx, maxy = geom_m.bounds
    width_m = maxx - minx
    height_m = maxy - miny
    
    width_px = width_m / resolution
    height_px = height_m / resolution

    width_tiles = int(np.ceil(width_px / dimension))
    height_tiles = int(np.ceil(height_px / dimension))

    tiles = np.zeros(shape=(height_tiles+1, width_tiles+1, 2))

    for i in range(height_tiles+1):
        for j in range(width_tiles+1):
            x = min(minx + j * dimension * resolution, maxx)
            y = min(miny + i * dimension * resolution, maxy)
            tiles[i, j] = [x, y]
                
    return tiles