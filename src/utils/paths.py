# src/utils/paths.py
from pathlib import Path

def get_project_root() -> Path:
    """Returns the root path of the project."""
    return Path(__file__).resolve().parents[2]

def get_data_path(filename: str = 'sampleForestPoly.geojson') -> Path:
    """Returns full path to a file in the DATA directory."""
    return get_project_root() / 'DATA' / filename