import pytest
from sentinelhub import BBox

from data_processing.geometry_processor import GeometryProcessor


@pytest.fixture
def geometry_processor() -> GeometryProcessor:
    return GeometryProcessor()


def test_extract_bbox_from_geometry():
    geometry = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [
                    [8.188143814749436, 48.61627297089052],
                    [8.39026475367633, 48.61627297089052],
                    [8.39026475367633, 48.542578737075878],
                    [8.188143814749436, 48.542578737075878],
                    [8.188143814749436, 48.61627297089052],
                ]
            ]
        ],
    }

    expected = BBox(
        (
            (911499.9999999999, 6197600),
            (934000, 6209999.999999998),
        ),
        crs="EPSG:3857",
    )

    result = GeometryProcessor.extract_bbox_from_geometry(
        geometry=geometry, geometry_crs="EPSG:4326", bbox_crs="EPSG:3857"
    )

    assert expected == result
