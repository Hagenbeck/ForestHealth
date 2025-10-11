import geojson
import numpy as np

from core.paths import get_data_path
from data_sourcing.sentinelhub_api import get_tiling_bounds


def test_get_tiling_bounds_one():
    geojson_path = get_data_path("sampleForestPoly.geojson")
    with open(geojson_path) as f:
        geo_file = geojson.load(f)

    geometry = geo_file["features"][0]["geometry"]

    expected = np.array(
        [
            [[887466.92179617, 6172045.9574239], [914510.99020897, 6172045.9574239]],
            [[887466.92179617, 6196382.7957982], [914510.99020897, 6196382.7957982]],
        ]
    )

    result = get_tiling_bounds(geometry)

    diff = result - expected
    diff_s = np.max(np.abs(diff))

    assert np.allclose(expected, result)
    assert diff_s < np.float64(1e-8)


def test_get_tiling_bounds_multiple():
    geojson_path = get_data_path("blackForestPoly.geojson")
    with open(geojson_path) as f:
        geo_file = geojson.load(f)

    geometry = geo_file["features"][0]["geometry"]

    expected = np.array(
        [
            [
                [837101.97665962, 6029135.7785605],
                [887101.97665962, 6029135.7785605],
                [937101.97665962, 6029135.7785605],
                [987101.97665962, 6029135.7785605],
                [1002586.52604665, 6029135.7785605],
            ],
            [
                [837101.97665962, 6079135.7785605],
                [887101.97665962, 6079135.7785605],
                [937101.97665962, 6079135.7785605],
                [987101.97665962, 6079135.7785605],
                [1002586.52604665, 6079135.7785605],
            ],
            [
                [837101.97665962, 6129135.7785605],
                [887101.97665962, 6129135.7785605],
                [937101.97665962, 6129135.7785605],
                [987101.97665962, 6129135.7785605],
                [1002586.52604665, 6129135.7785605],
            ],
            [
                [837101.97665962, 6179135.7785605],
                [887101.97665962, 6179135.7785605],
                [937101.97665962, 6179135.7785605],
                [987101.97665962, 6179135.7785605],
                [1002586.52604665, 6179135.7785605],
            ],
            [
                [837101.97665962, 6229135.7785605],
                [887101.97665962, 6229135.7785605],
                [937101.97665962, 6229135.7785605],
                [987101.97665962, 6229135.7785605],
                [1002586.52604665, 6229135.7785605],
            ],
            [
                [837101.97665962, 6272801.91063193],
                [887101.97665962, 6272801.91063193],
                [937101.97665962, 6272801.91063193],
                [987101.97665962, 6272801.91063193],
                [1002586.52604665, 6272801.91063193],
            ],
        ]
    )

    result = get_tiling_bounds(geometry)

    diff = result - expected
    diff_s = np.max(np.abs(diff))

    assert np.allclose(expected, result)
    assert diff_s < np.float64(1e-8)
