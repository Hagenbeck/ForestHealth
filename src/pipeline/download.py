import json
import time
from datetime import datetime
from typing import Optional

import numpy as np
from rasterio.io import MemoryFile
from requests import Response

import config as cf
from core.date_utils import generate_monthly_interval, parse_date
from core.paths import get_data_path
from data_sourcing.data_models import EvalScriptType
from data_sourcing.geometry_toolkit import GeometryToolkit
from data_sourcing.sentinelhub_api import SentinelHubAPI


class DownloadPipeline:
    """
    Encapsulates sentinel download logic. Instantiate with GeometryToolkit and SentinelHubAPI
    (or let the class create defaults). Call run() to get the stacked array for all intervals.
    """

    def __init__(
        self,
        geom_tools: Optional[GeometryToolkit] = None,
        sentinel_api: Optional[SentinelHubAPI] = None,
        config_module=cf,
    ):
        self.config = config_module
        self.geom_tools = geom_tools or GeometryToolkit(
            self.config.GEOMETRY_FILE, self.config.GEOMETRY_FILE_CRS
        )
        self.sentinel_api = sentinel_api or SentinelHubAPI()

    def run(self) -> np.ndarray:
        start_date = parse_date(self.config.START_DATE)
        end_date = parse_date(self.config.END_DATE)

        ms_date, me_date = generate_monthly_interval(
            start_date=start_date, end_date=end_date
        )

        result = []
        for ind, start_interval in enumerate(ms_date):
            data = self.request_and_stack_tiles(
                evalscript_type=self.config.EVALSCRIPT_TYPE,
                start_date=start_interval,
                end_date=me_date[ind],
            )
            result.append(data)

        np.save(get_data_path(cf.OBSERVATION_SAVE_FILE), result)
        return np.array(result)

    @staticmethod
    def validate_response_content(response: Response) -> bool:
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(
            f"Response content type: {response.headers.get('content-type', 'Unknown')}"
        )
        print(f"Response content length: {len(response.content)} bytes")

        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                error_data = response.json()
                print(f"API returned JSON error: {json.dumps(error_data, indent=2)}")
                return False
            except Exception:
                pass

        if len(response.content) < 1000:
            print(f"Response content is very small ({len(response.content)} bytes)")
            print(f"First 200 bytes: {response.content[:200]}")
            return False

        return True

    def request_and_stack_tiles(
        self,
        evalscript_type: EvalScriptType,
        start_date: datetime,
        end_date: datetime,
    ) -> np.ndarray:
        height, width, coords = np.shape(self.geom_tools.tiles)

        sentinelhub_api = self.sentinel_api

        tile_data_grid = []
        bands = None

        for i in range(height - 2, -1, -1):
            row_tiles = []
            for j in range(width - 1):
                bbox = self.geom_tools.get_bbox(i, j)
                width_px, height_px = self.geom_tools.get_pixels(bbox)

                if not self.geom_tools.bbox_intersects_geometry(bbox):
                    row_tiles.append(
                        {"data": None, "width_px": width_px, "height_px": height_px}
                    )
                else:
                    json_request = sentinelhub_api.build_json_request(
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
                        response = sentinelhub_api.safe_send_request()

                        if not self.validate_response_content(response):
                            print(
                                f"Invalid response for tile [{i}, {j}], filling with zeros"
                            )
                            row_tiles.append(
                                {
                                    "data": None,
                                    "width_px": width_px,
                                    "height_px": height_px,
                                }
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
                            debug_filename = f"debug_response_tile_{i}_{j}.bin"
                            with open(debug_filename, "wb") as f:
                                f.write(response.content)
                            print(f"   Saved response content to {debug_filename}")

                            row_tiles.append(
                                {
                                    "data": None,
                                    "width_px": width_px,
                                    "height_px": height_px,
                                }
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
