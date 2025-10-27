from datetime import datetime, timedelta

import pandas as pd

from core.date_utils import (
    generate_july_intervals,
    generate_monthly_interval,
    parse_date,
)


def test_parse_date():
    assert abs(datetime.now() - parse_date("now")) < timedelta(seconds=1)
    assert datetime(2024, 11, 23) == parse_date("2024-11-23")


def test_generate_july_interval():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 9, 13)

    dr_start, dr_end = generate_july_intervals(start_date=start_date, end_date=end_date)

    expected_start = [datetime(2023, 7, 1), datetime(2024, 7, 1), datetime(2025, 7, 1)]
    expected_end = [datetime(2023, 7, 31), datetime(2024, 7, 31), datetime(2025, 7, 31)]

    assert dr_start == expected_start
    assert dr_end == expected_end


def test_generate_july_interval_one():
    start_date = datetime(2025, 7, 1)
    end_date = datetime(2025, 7, 31)

    dr_start, dr_end = generate_july_intervals(start_date=start_date, end_date=end_date)

    expected_start = [datetime(2025, 7, 1)]
    expected_end = [datetime(2025, 7, 31)]

    assert dr_start == expected_start
    assert dr_end == expected_end


def test_generate_monthly_interval():
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 6, 30)

    dr_start, dr_end = generate_monthly_interval(
        start_date=start_date,
        end_date=end_date,
    )

    expected_start = pd.DatetimeIndex(
        [
            "2025-01-01",
            "2025-02-01",
            "2025-03-01",
            "2025-04-01",
            "2025-05-01",
            "2025-06-01",
        ],
        dtype="datetime64[ns]",
        freq="MS",
    )

    expected_end = pd.DatetimeIndex(
        [
            "2025-01-31",
            "2025-02-28",
            "2025-03-31",
            "2025-04-30",
            "2025-05-31",
            "2025-06-30",
        ],
        dtype="datetime64[ns]",
        freq="ME",
    )

    assert (dr_start == expected_start).all()
    assert (dr_end == expected_end).all()
