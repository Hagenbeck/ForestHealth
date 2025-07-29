from pandas import DatetimeIndex
from src.utils.date_helper import *
from datetime import datetime, timedelta

def test_parse_date():
    assert abs(datetime.now() - parse_date("now")) < timedelta(seconds=1)
    assert datetime(2024, 11, 23) == parse_date("2024-11-23")
    
def test_generate_monthly_interval():
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    
    dr_start, dr_end = generate_monthly_interval(start_date=start_date, end_date=end_date)
    
    expected_start = DatetimeIndex(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01'],
              dtype='datetime64[ns]', freq='MS')
    expected_end = DatetimeIndex(['2024-01-31', '2024-02-29', '2024-03-31', '2024-04-30', '2024-05-31', '2024-06-30'],
              dtype='datetime64[ns]', freq='ME')

    
    assert all(dr_start == expected_start)
    assert all(dr_end == expected_end)
    
def test_generate_monthly_interval_one():
    start_date = datetime(2025, 7, 1)
    end_date = datetime(2025, 7, 31)
    
    dr_start, dr_end = generate_monthly_interval(start_date=start_date, end_date=end_date)
    
    expected_start = DatetimeIndex(['2025-07-01'],
              dtype='datetime64[ns]', freq='MS')
    expected_end = DatetimeIndex(['2025-07-31'],
              dtype='datetime64[ns]', freq='ME')

    
    assert all(dr_start == expected_start)
    assert all(dr_end == expected_end)