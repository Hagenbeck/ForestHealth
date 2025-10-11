from datetime import datetime, timedelta

from core.date_utils import generate_july_intervals, parse_date


def test_parse_date():
    assert abs(datetime.now() - parse_date("now")) < timedelta(seconds=1)
    assert datetime(2024, 11, 23) == parse_date("2024-11-23")


def test_generate_monthly_interval():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 9, 13)

    dr_start, dr_end = generate_july_intervals(start_date=start_date, end_date=end_date)

    expected_start = [datetime(2023, 7, 1), datetime(2024, 7, 1), datetime(2025, 7, 1)]
    expected_end = [datetime(2023, 7, 31), datetime(2024, 7, 31), datetime(2025, 7, 31)]

    assert dr_start == expected_start
    assert dr_end == expected_end


def test_generate_monthly_interval_one():
    start_date = datetime(2025, 7, 1)
    end_date = datetime(2025, 7, 31)

    dr_start, dr_end = generate_july_intervals(start_date=start_date, end_date=end_date)

    expected_start = [datetime(2025, 7, 1)]
    expected_end = [datetime(2025, 7, 31)]

    assert dr_start == expected_start
    assert dr_end == expected_end
