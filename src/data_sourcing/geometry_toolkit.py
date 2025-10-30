import geojson
import numpy as np
from pyproj import Transformer
from sentinelhub import CRS, BBox
from shapely import Geometry
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from core.paths import get_data_path
from data_sourcing.data_models import CRSType


class GeometryToolkit:
    aoi_geometry: dict
    aoi_geometry_shape: Geometry
    tiles: np.ndarray
    resolution: int
    max_dimension: int
    aoi_crs: CRSType

    def __init__(
        self,
        aoi_file: str,
        aoi_crs: CRSType,
        resolution: int = 20,
        max_dimension: int = 2500,
    ):
        self.resolution = resolution
        self.max_dimension = max_dimension
        self.aoi_crs = aoi_crs

        geojson_path = get_data_path(aoi_file)
        self.aoi_geometry = self.retrieve_geometry(geojson_path)
        self.aoi_geometry_shape = shape(self.aoi_geometry)

        self._geometry_3857: BaseGeometry | None = None
        self.get_tiling_bounds()

    def retrieve_geometry(self, geojson_path: str) -> dict:
        """
        Retrieve the geometry from a geojson file

        Args:
            geojson_path (str): Path to the .geojson

        Returns:
            dict: geometry of geojson
        """
        with open(geojson_path) as f:
            geo_file = geojson.load(f)

        return geo_file["features"][0]["geometry"]

    def get_tiling_bounds(self) -> np.ndarray:
        """
        Calculates the tiles needed to fetch data from the sentinelhub API at the highest resolution.

        Returns
        -------
        np.ndarray
            Array with the corners of all tiles
        """

        project = Transformer.from_crs(
            self.aoi_crs, "EPSG:3857", always_xy=True
        ).transform
        geom_m = transform(project, self.aoi_geometry_shape)

        minx, miny, maxx, maxy = geom_m.bounds
        width_m = maxx - minx
        height_m = maxy - miny

        width_px = width_m / self.resolution
        height_px = height_m / self.resolution

        width_tiles = int(np.ceil(width_px / self.max_dimension))
        height_tiles = int(np.ceil(height_px / self.max_dimension))

        tiles = np.zeros(shape=(height_tiles + 1, width_tiles + 1, 2))

        for i in range(height_tiles + 1):
            for j in range(width_tiles + 1):
                x = min(minx + j * self.max_dimension * self.resolution, maxx)
                y = min(miny + i * self.max_dimension * self.resolution, maxy)
                tiles[i, j] = [x, y]

        self.tiles = tiles
        return tiles

    def get_geometry_as_3857(self) -> BaseGeometry:
        """
        Transform a geometry from EPSG:4326 to EPSG:3857.
        Use this to pre-transform geometry for better performance.
        """
        if self._geometry_3857 is None:
            transformer = Transformer.from_crs(
                self.aoi_crs, "EPSG:3857", always_xy=True
            )
            self._geometry_3857 = transform(
                transformer.transform, self.aoi_geometry_shape
            )
        return self._geometry_3857

    def bbox_intersects_geometry(self, bbox: BBox) -> bool:
        """
        Check if a bbox intersects with a geometry, handling CRS transformations.

        Parameters:
        -----------
        bbox : BBox
            BBox object in Web Mercator (EPSG:3857)

        Returns:
        --------
        bool
            True if bbox intersects geometry, False otherwise
        """
        bbox_geom = box(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y)

        geometry_3857 = self.get_geometry_as_3857()

        return bbox_geom.intersects(geometry_3857)

    def get_bbox(self, y: int, x: int) -> BBox:
        """
        Calculates the bounding box for a tile of a split request

        Args:
            y (int): y index of tiles
            x (int): x index of tiles

        Returns:
            BBox: Bounding Box of tile at y, x
        """
        tile_coords = np.array(
            [
                [self.tiles[y, x], self.tiles[y, x + 1]],
                [self.tiles[y + 1, x], self.tiles[y + 1, x + 1]],
            ]
        )

        flat_coords = tile_coords.reshape(-1, 2)
        xs = flat_coords[:, 0]
        ys = flat_coords[:, 1]

        return BBox(bbox=[xs.min(), ys.min(), xs.max(), ys.max()], crs=CRS.POP_WEB)

    def get_pixels(self, bbox: BBox) -> tuple[int, int]:
        """
        Calculate the width and height of a bbox in pixels

        Args:
            bbox (BBox): bounding Box

        Returns:
            tuple[int, int]: width, height of bbox in pixels
        """

        width_m = bbox.max_x - bbox.min_x
        height_m = bbox.max_y - bbox.min_y

        width_px = int(width_m / self.resolution)
        height_px = int(height_m / self.resolution)

        return width_px, height_px
