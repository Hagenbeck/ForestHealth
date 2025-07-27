import src.config as conf

from src.utils.date_helper import parse_date
from src.utils.evalscripts import get_evalscript, get_response_setup
from src.data_models import EvalScriptType

def build_json_request(width_px: int, height_px: int, evalscript_type: EvalScriptType = "RGB", bbox: list[float] | None = None, geometry: dict | None = None) -> dict:
    start_date = parse_date(conf.START_DATE)
    end_date = parse_date(conf.END_DATE)
    
    evalscript = get_evalscript(evalscript_type)
    responses = get_response_setup(evalscript_type)
    
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
                                'dataFilter': {
                                    'timeRange': {
                                        'from': f'{start_date}T00:00:00Z',
                                        'to': f'{end_date}T23:59:59Z'
                                    }
                                },
                                'processing': {
                                    'mosaicking': 'ORBIT'
                                }
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