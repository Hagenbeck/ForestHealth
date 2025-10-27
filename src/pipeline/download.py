import json
import os
import time
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from oauthlib.oauth2 import BackendApplicationClient
from rasterio.io import MemoryFile
from requests import Response
from requests_oauthlib import OAuth2Session

import core.config as cf
from core.date_utils import generate_monthly_interval, parse_date
from core.geometry import (
    bbox_intersects_geometry,
    get_bbox,
    get_pixels,
    retrieve_geometry,
    transform_geometry_to_3857,
)
from core.paths import get_data_path
from data_sourcing.data_models import CRSType, EvalScriptType
from data_sourcing.sentinelhub_api import (
    build_json_request,
    get_tiling_bounds,
    safe_send_request,
)


def download_sentinel_data(
    geojson: str = "aoi_hornisgrinde.geojson",
    evalscript_type: EvalScriptType = "INDICES",
    crs_type: CRSType = "EPSG:4326",
) -> np.array:
    """
    Download the sentinel data for a given geojson with specified evalscripts

    Args:
        geojson (str, optional): Path to geojson file with AOI boundaries. Defaults to "aoi_hornisgrinde.geojson".
        evalscript_type (EvalScriptType, optional): Type of request, influences evalscripts. Defaults to "INDICES".

    Returns:
        np.array: array with sentinel data for specified AOI
    """

    geojson_path = get_data_path(geojson)
    geometry = retrieve_geometry(geojson_path)

    tiles = get_tiling_bounds(geometry=geometry, crs=crs_type)

    start_date = parse_date(cf.START_DATE)
    end_date = parse_date(cf.END_DATE)

    ms_date, me_date = generate_monthly_interval(
        start_date=start_date, end_date=end_date
    )

    result = []
    for ind, start_interval in enumerate(ms_date):
        data = request_and_stack_tiles(
            tiles,
            geometry,
            evalscript_type=evalscript_type,
            start_date=start_interval,
            end_date=me_date[ind],
            crs=crs_type,
        )
        result.append(data)

    return np.array(result)


def retrieve_secrets() -> tuple[OAuth2Session, str, str]:
    """
    Load and initialize os environment variables

    Raises:
        EnvironmentError: if the sentinelhub credentials are not specified

    Returns:
        tuple[OAuth2Session, str, str]: returns the OAuth Session, the secret and token_url
    """
    load_dotenv()

    client_id = os.getenv("SENTINELHUB_CLIENT_ID")
    client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")
    token_url = os.getenv(
        "SENTINELHUB_TOKEN_URL",
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    )

    if not client_id or not client_secret:
        raise EnvironmentError(
            "Missing SENTINELHUB_CLIENT_ID or SENTINELHUB_CLIENT_SECRET in .env"
        )

    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)

    return oauth, client_secret, token_url


def validate_response_content(response: Response) -> bool:
    """
    Validating the reponse to contain a valid image

    Args:
        response (Response): response of the sentinelhub-api

    Returns:
        bool: True if the response likely contains a valid image
    """
    print(f"Response status: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    print(f"Response content type: {response.headers.get('content-type', 'Unknown')}")
    print(f"Response content length: {len(response.content)} bytes")

    # Check if response is JSON (error response)
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            error_data = response.json()
            print(f"API returned JSON error: {json.dumps(error_data, indent=2)}")
            return False
        except Exception:
            pass

    # Check if content is too small to be a valid image
    if len(response.content) < 1000:
        print(f"Response content is very small ({len(response.content)} bytes)")
        print(f"First 200 bytes: {response.content[:200]}")
        return False

    return True


def request_and_stack_tiles(
    tiles: np.ndarray,
    geometry: dict,
    evalscript_type: EvalScriptType,
    start_date: datetime,
    end_date: datetime,
    crs: CRSType,
) -> np.array:
    """
    Requesting and tiling all the tiles for a given time period in a given AOI

    Args:
        tiles (np.ndarray): Tiled bounds of AOI
        geometry (dict): geometry as bounds
        evalscript_type (EvalScriptType): evalscript Type to specify the Javascript Section
        start_date (datetime): start date of consideration interval
        end_date (datetime): end date of consideration interval

    Returns:
        np.array: sentinel data of AOI for the specified time period
    """

    oauth, client_secret, token_url = retrieve_secrets()

    height, width, coords = np.shape(tiles)

    geometry_3857 = transform_geometry_to_3857(geometry, crs=crs)

    tile_data_grid = []
    bands = None

    for i in range(height - 2, -1, -1):
        row_tiles = []
        for j in range(width - 1):
            bbox = get_bbox(i, j, tiles)
            width_px, height_px = get_pixels(bbox)

            if not bbox_intersects_geometry(bbox, geometry_3857=geometry_3857, crs=crs):
                row_tiles.append(
                    {"data": None, "width_px": width_px, "height_px": height_px}
                )
            else:
                json_request = build_json_request(
                    width_px=width_px,
                    height_px=height_px,
                    start_date=start_date,
                    end_date=end_date,
                    evalscript_type=evalscript_type,
                    bbox=list(bbox),
                    crs="EPSG:3857",
                )

                print(f"Sending request for tile [{i}, {j}] with:")
                print("BBox:", list(bbox))
                print(f"width: {width_px}, height: {height_px}")
                print(
                    "Date range:",
                    json_request["input"]["data"][0]["dataFilter"]["timeRange"],
                )

                try:
                    response = safe_send_request(
                        client_secret=client_secret,
                        token_url=token_url,
                        oauth=oauth,
                        json_request=json_request,
                    )

                    if not validate_response_content(response):
                        print(
                            f"Invalid response for tile [{i}, {j}], filling with zeros"
                        )
                        row_tiles.append(
                            {"data": None, "width_px": width_px, "height_px": height_px}
                        )
                        continue

                    try:
                        with MemoryFile(response.content) as memfile:
                            with memfile.open() as ds:
                                tile_data = ds.read()
                                print(
                                    f"Successfully read tile [{i}, {j}]: shape {tile_data.shape}"
                                )

                                if bands is None:
                                    bands = tile_data.shape[0]
                                    print(f"   Detected {bands} bands")

                                row_tiles.append(
                                    {
                                        "data": tile_data,
                                        "width_px": width_px,
                                        "height_px": height_px,
                                    }
                                )

                    except Exception as raster_error:
                        print(
                            f"Failed to read tile [{i}, {j}] as raster: {raster_error}"
                        )
                        print("Saving problematic response to debug file...")

                        debug_filename = f"debug_response_tile_{i}_{j}.bin"
                        with open(debug_filename, "wb") as f:
                            f.write(response.content)
                        print(f"   Saved response content to {debug_filename}")

                        row_tiles.append(
                            {"data": None, "width_px": width_px, "height_px": height_px}
                        )

                except Exception as request_error:
                    print(f"Request failed for tile [{i}, {j}]: {request_error}")
                    row_tiles.append(
                        {"data": None, "width_px": width_px, "height_px": height_px}
                    )

            time.sleep(1)

        tile_data_grid.append(row_tiles)

    if bands is None:
        raise ValueError("No valid tiles found in the geometry")

    stacked_rows = []
    for row in tile_data_grid:
        row_data = []
        for tile_info in row:
            if tile_info["data"] is None:
                zeros_tile = np.zeros(
                    (bands, tile_info["height_px"], tile_info["width_px"])
                )
                row_data.append(zeros_tile)
            else:
                row_data.append(tile_info["data"])

        stacked_row = np.concatenate(row_data, axis=2)
        stacked_rows.append(stacked_row)

    final_data = np.concatenate(stacked_rows, axis=1)

    return final_data
