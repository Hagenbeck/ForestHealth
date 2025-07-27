from datetime import datetime
import pandas as pd

def parse_date(date: str):
    return datetime.now() if date == "now" else datetime.strptime(date, "%Y-%m-%d")

def generate_monthly_interval(start_date: datetime, end_date: datetime):
    return pd.date_range(start=start_date, end=end_date, freq="MS"), pd.date_range(start=start_date, end=end_date, freq="ME")