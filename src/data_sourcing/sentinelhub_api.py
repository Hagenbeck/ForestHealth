import os
import time
from datetime import datetime

from dotenv import load_dotenv
from oauthlib.oauth2 import BackendApplicationClient
from requests import HTTPError, Response
from requests.exceptions import RequestException
from requests_oauthlib import OAuth2Session

import config as conf
from data_sourcing.data_models import CRSType, EvalScriptType
from data_sourcing.evalscripts import get_evalscript, get_response_setup


class SentinelHubAPI:
    def __init__(self):
        self.retrieve_secrets()
        self.json_request = None

    def retrieve_secrets(self):
        """
        Load and initialize os environment variables

        Raises:
            EnvironmentError: if the sentinelhub credentials are not specified

        Returns:
            tuple[OAuth2Session, str, str]: returns the OAuth Session, the secret and token_url
        """
        load_dotenv()

        client_id = os.getenv("SENTINELHUB_CLIENT_ID")
        self.client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")
        self.token_url = os.getenv(
            "SENTINELHUB_TOKEN_URL",
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        )

        if not client_id or not self.client_secret:
            raise EnvironmentError(
                "Missing SENTINELHUB_CLIENT_ID or SENTINELHUB_CLIENT_SECRET in .env"
            )

        client = BackendApplicationClient(client_id=client_id)
        self.oauth = OAuth2Session(client=client)
        self.client_secret
        self.token_url

    def build_json_request(
        self,
        width_px: int,
        height_px: int,
        start_date: datetime,
        end_date: datetime,
        evalscript_type: EvalScriptType = "RGB",
        bbox: list[float] | None = None,
        geometry: dict | None = None,
        crs: CRSType = "EPSG:3857",
    ) -> dict:
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
            raise ValueError(
                f"The API allows for a maximum of 2500 pixels. {width_px} is too wide."
            )
        elif height_px > 2500:
            raise ValueError(
                f"The API allows for a maximum of 2500 pixels. {height_px} is too high."
            )

        evalscript = get_evalscript(evalscript_type)
        responses = get_response_setup(evalscript_type)

        if evalscript_type == "INDICES":
            processing_block = {"mosaicking": "ORBIT"}
            data_filter = {
                "timeRange": {
                    "from": f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z",
                    "to": f"{end_date.strftime('%Y-%m-%d')}T23:59:59Z",
                }
            }
        else:
            processing_block = {}
            data_filter = {
                "timeRange": {
                    "from": f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z",
                    "to": f"{end_date.strftime('%Y-%m-%d')}T23:59:59Z",
                },
                "mosaickingOrder": "leastCC",
                "maxCloudCoverage": 20,
            }

        json_request = {
            "input": {
                "bounds": {"properties": {}},
                "data": [
                    {
                        "type": conf.COLLECTION_ID.upper(),
                        "dataFilter": data_filter,
                        "processing": processing_block,
                    }
                ],
            },
            "output": {"width": width_px, "height": height_px, "responses": responses},
            "evalscript": evalscript,
        }

        if crs == "EPSG:3857":
            json_request["input"]["bounds"]["properties"]["crs"] = (
                "http://www.opengis.net/def/crs/EPSG/0/3857"
            )
        elif crs == "CRS84":
            json_request["input"]["bounds"]["properties"]["crs"] = (
                "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
            )

        if bbox is None and geometry is None:
            raise ValueError("Either 'bbox' or 'geometry' must be provided.")
        elif bbox is not None:
            json_request["input"]["bounds"]["bbox"] = bbox
        else:
            json_request["input"]["bounds"]["geometry"] = geometry

        self.json_request = json_request
        return json_request

    def send_request(
        self,
    ) -> Response:
        """
        Sends the request to the sentinel hub process API.

        Parameters
        ----------
        client_secret : str
            The secret to retrieve the access token
        token_url : str
            The url where the token should be retrieved
        oauth: OAuth2Session
            The OAuthSession to retrieve the token
        json_request: dict
            The request in JSON format

        Returns
        -------
        Response
            Response of the sentinel hub process API
        """
        if self.json_request is None:
            raise ValueError("JSON-Request was not built")

        token = self.oauth.fetch_token(
            token_url=self.token_url,
            client_secret=self.client_secret,
            include_client_id=True,
        )

        url_request = "https://sh.dataspace.copernicus.eu/api/v1/process"
        headers_request = {
            "Authorization": f"Bearer {token['access_token']}",
            "Content-Type": "application/json",
        }

        response = self.oauth.post(
            url_request, headers=headers_request, json=self.json_request
        )

        return response

    def safe_send_request(self, max_retries=3) -> Response:
        """
        Safely sends a request with retry logic for rate limits and server errors.

        Parameters
        ----------
        client_secret : str
            The secret to retrieve the access token
        token_url : str
            The url where the token should be retrieved
        oauth: OAuth2Session
            The OAuthSession to retrieve the token
        json_request: dict
            The request in JSON format
        max_retries: int
            Maximum number of retry attempts

        Returns
        -------
        Response
            Successful response from the API

        Raises
        ------
        HTTPError
            If the request fails after all retries
        RuntimeError
            If maximum retries exceeded
        """
        retries = 0

        while retries < max_retries:
            try:
                response = self.send_request()

                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    retry_after_ms = response.headers.get("retry-after", "2000")
                    try:
                        wait_time_ms = int(retry_after_ms)
                        wait_time_sec = wait_time_ms / 1000.0
                    except (ValueError, TypeError):
                        wait_time_sec = 2.0

                    print(
                        f"Rate limit hit (attempt {retries + 1}/{max_retries}). Waiting {wait_time_sec:.1f} seconds..."
                    )

                    time.sleep(wait_time_sec)
                    retries += 1

                elif response.status_code in [500, 502, 503, 504]:
                    wait_time = min(2**retries, 16)

                    print(
                        f"Server error {response.status_code} (attempt {retries + 1}/{max_retries}). Waiting {wait_time} seconds..."
                    )
                    time.sleep(wait_time)

                    retries += 1

                else:
                    error_msg = (
                        f"Request failed with status code {response.status_code}"
                    )
                    try:
                        error_details = response.json()
                        error_msg += f": {error_details}"
                    except Exception:
                        error_msg += f": {response.text}"

                    raise HTTPError(error_msg, response=response)

            except RequestException as e:
                if retries < max_retries - 1:
                    wait_time = min(2**retries, 8)
                    print(
                        f"Network error (attempt {retries + 1}/{max_retries}): {e}. Waiting {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    retries += 1
                else:
                    raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise

        raise RuntimeError(
            f"Failed after {max_retries} retries due to rate limits or server errors."
        )
