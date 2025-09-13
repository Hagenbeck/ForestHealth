from datetime import datetime
import pandas as pd

def parse_date(date: str):
    '''Converts a valid date string into a datetime object'''
    return datetime.now() if date == "now" else datetime.strptime(date, "%Y-%m-%d")

def generate_july_intervals(start_date: datetime, end_date: datetime):
    '''Returns two lists of datetime objects representing July 1st and July 31st for each year in the interval'''
    years = range(start_date.year, end_date.year+1)
    july_starts = [datetime(year, 7, 1) for year in years]
    july_ends = [datetime(year, 7, 31) for year in years]
    return july_starts, july_ends