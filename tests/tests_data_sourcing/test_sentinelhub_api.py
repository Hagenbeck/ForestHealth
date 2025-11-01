from unittest.mock import Mock

import pytest
from requests import HTTPError

from data_sourcing.sentinelhub_api import SentinelHubAPI


def make_response(status_code=200, headers=None, json_data=None, text="", content=b""):
    resp = Mock(spec=["status_code", "headers", "json", "text", "content"])
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text
    resp.content = content

    if json_data is None:
        # default json() raises to simulate non-json body
        resp.json = Mock(side_effect=ValueError("No JSON"))
    else:
        resp.json = Mock(return_value=json_data)

    return resp


@pytest.fixture(autouse=True)
def disable_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    yield


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(SentinelHubAPI, "retrieve_secrets", lambda self: None)
    instance = SentinelHubAPI()
    instance.json_request = {}
    return instance


def test_safe_send_request_immediate_success(api):
    resp200 = make_response(status_code=200)
    api.send_request = Mock(return_value=resp200)

    out = api.safe_send_request(max_retries=3)
    assert out is resp200
    assert api.send_request.call_count == 1


def test_safe_send_request_rate_limit_then_success(api):
    resp429 = make_response(status_code=429, headers={"retry-after": "3000"})
    resp200 = make_response(status_code=200)
    api.send_request = Mock(side_effect=[resp429, resp200])

    out = api.safe_send_request(max_retries=3)
    assert out is resp200
    assert api.send_request.call_count == 2


def test_safe_send_request_rate_limit_non_numeric_retry_header(api):
    resp429 = make_response(status_code=429, headers={"retry-after": "not-a-number"})
    resp200 = make_response(status_code=200)
    api.send_request = Mock(side_effect=[resp429, resp200])

    out = api.safe_send_request(max_retries=3)
    assert out is resp200
    assert api.send_request.call_count == 2


def test_safe_send_request_server_error_then_success(api):
    resp500 = make_response(status_code=500, json_data={"error": "server"})
    resp200 = make_response(status_code=200)
    api.send_request = Mock(side_effect=[resp500, resp200])

    out = api.safe_send_request(max_retries=3)
    assert out is resp200
    assert api.send_request.call_count == 2


def test_safe_send_request_other_error_raises(api):
    resp400 = make_response(status_code=400, text="Bad Request")
    api.send_request = Mock(return_value=resp400)

    with pytest.raises(HTTPError):
        api.safe_send_request(max_retries=1)
    assert api.send_request.call_count == 1
