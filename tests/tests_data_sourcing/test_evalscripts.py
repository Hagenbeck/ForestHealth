from data_sourcing.evalscripts import (
    all_bands_evalscript,
    all_bands_response,
    get_evalscript,
    get_response_setup,
    indices_evalscript,
    indices_response,
    rgb_evalscript,
    rgb_response,
)


def test_get_evalscript():
    assert rgb_evalscript == get_evalscript("RGB")
    assert all_bands_evalscript == get_evalscript("ALL")
    assert indices_evalscript == get_evalscript("INDICES")


def test_get_response_setup():
    assert rgb_response == get_response_setup("RGB")
    assert all_bands_response == get_response_setup("ALL")
    assert indices_response == get_response_setup("INDICES")
