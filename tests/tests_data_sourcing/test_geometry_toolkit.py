import numpy as np
import pytest
from sentinelhub import CRS, BBox

from data_sourcing.geometry_toolkit import GeometryToolkit


@pytest.fixture
def geometry_toolkit() -> GeometryToolkit:
    return GeometryToolkit("sampleForestPoly.geojson", aoi_crs="EPSG:4326")


def test_get_tiling_bounds_one(geometry_toolkit: GeometryToolkit):
    expected = np.array(
        [
            [[887466.92179617, 6172045.9574239], [914510.99020897, 6172045.9574239]],
            [[887466.92179617, 6196382.7957982], [914510.99020897, 6196382.7957982]],
        ]
    )

    result = geometry_toolkit.get_tiling_bounds()

    diff = result - expected
    diff_s = np.max(np.abs(diff))

    assert np.allclose(expected, result)
    assert diff_s < np.float64(1e-8)


def test_bbox_pixels_and_intersection(geometry_toolkit: GeometryToolkit):
    # pick the first tile bbox
    bbox = geometry_toolkit.get_bbox(0, 0)

    # bbox should intersect the AOI
    assert geometry_toolkit.bbox_intersects_geometry(bbox) is True

    # pixels computed by service should match manual calculation
    width_px, height_px = geometry_toolkit.get_pixels(bbox)
    expected_width = int((bbox.max_x - bbox.min_x) / geometry_toolkit.resolution)
    expected_height = int((bbox.max_y - bbox.min_y) / geometry_toolkit.resolution)
    assert width_px == expected_width
    assert height_px == expected_height


def test_non_intersecting_bbox(geometry_toolkit: GeometryToolkit):
    # construct a bbox shifted far away so it does not intersect AOI
    tiles = geometry_toolkit.tiles
    # take a corner and shift it by 1e6 meters
    shift = 1_000_000.0
    x0, y0 = float(tiles[0, 0, 0]), float(tiles[0, 0, 1])
    bbox_out = BBox(
        bbox=[x0 + shift, y0 + shift, x0 + shift + 1000.0, y0 + shift + 1000.0],
        crs=CRS.POP_WEB,
    )

    assert geometry_toolkit.bbox_intersects_geometry(bbox_out) is False


def test_geometry_bounds_consistent_with_tiles(geometry_toolkit: GeometryToolkit):
    geom_3857 = geometry_toolkit.get_geometry_as_3857()
    gminx, gminy, gmaxx, gmaxy = geom_3857.bounds

    tiles = geometry_toolkit.tiles
    # tiles store corner coordinates; compute min/max from tiles array
    xs = tiles[..., 0].ravel()
    ys = tiles[..., 1].ravel()
    tminx, tminy, tmaxx, tmaxy = (
        float(xs.min()),
        float(ys.min()),
        float(xs.max()),
        float(ys.max()),
    )

    # geometry bounds should lie within or equal to the tiles min/max (allow tiny numerical tolerance)
    assert np.isclose(gminx, tminx, atol=1e-6) or (gminx >= tminx - 1e-6)
    assert np.isclose(gminy, tminy, atol=1e-6) or (gminy >= tminy - 1e-6)
    assert np.isclose(gmaxx, tmaxx, atol=1e-6) or (gmaxx <= tmaxx + 1e-6)
    assert np.isclose(gmaxy, tmaxy, atol=1e-6) or (gmaxy <= tmaxy + 1e-6)


def test_tiles_monotonic_and_shape(geometry_toolkit: GeometryToolkit):
    tiles = geometry_toolkit.tiles
    # tiles should be 3D array with last dim = 2 (x,y)
    assert tiles.ndim == 3
    assert tiles.shape[2] == 2

    # x should be non-decreasing along columns, y non-decreasing along rows
    xs = tiles[..., 0]
    ys = tiles[..., 1]
    # check monotonic increase across columns (each row)
    assert np.all(np.diff(xs, axis=1) >= 0)
    # check monotonic increase across rows (each column)
    assert np.all(np.diff(ys, axis=0) >= 0)
