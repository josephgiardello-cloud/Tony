import argparse
import os
from datetime import datetime
from typing import List

def parse_years(years_str: str) -> List[int]:
    """Validate and parse comma-separated years."""
    try:
        years = [int(y.strip()) for y in years_str.split(",")]
        for y in years:
            if y < 1900 or y > datetime.now().year:
                raise ValueError(f"Invalid year: {y}")
        return years
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))

def file_exists(path: str) -> bool:
    """Check if a file exists."""
    return os.path.exists(path)
