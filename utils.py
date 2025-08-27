utils.py

import os
import glob
from typing import List
from google.colab import drive
from .config import SETTINGS

def mount_google_drive():
    """Mounts Google Drive if not already mounted."""
    mount_point = '/content/drive'
    if not os.path.isdir(mount_point):
        print("Mounting Google Drive...")
        drive.mount(mount_point)
        print("✅ Google Drive mounted successfully.")
    else:
        print("✅ Google Drive is already mounted.")

def setup_directories():
    """Creates all necessary directories defined in the configuration."""
    print("--- Setting up project directories ---")
    paths_to_create = [
        SETTINGS.BASE_PATH,
        SETTINGS.get_raw_trades_path("in_sample"),
        SETTINGS.get_raw_trades_path("out_of_sample"),
        SETTINGS.get_processed_trades_path("in_sample"),
        SETTINGS.get_processed_trades_path("out_of_sample"),
        SETTINGS.get_funding_rate_path()
    ]
    for path in paths_to_create:
        os.makedirs(path, exist_ok=True)
        print(f"Directory ensured: {path}")
    print("✅ All directories are set up.")

def get_files_for_period(period_name: str, data_type: str = "processed_trades") -> List[str]:
    """
    Gets a sorted list of Parquet files for a given period and data type.
    """
    if data_type == "processed_trades":
        path_pattern = os.path.join(SETTINGS.get_processed_trades_path(period_name), "*.parquet")
    elif data_type == "raw_trades":
         path_pattern = os.path.join(SETTINGS.get_raw_trades_path(period_name), f"*{SETTINGS.ASSET}*.zip")
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    files = sorted(glob.glob(path_pattern))
    if not files:
        print(f"⚠️ Warning: No files found for period '{period_name}' with data type '{data_type}'.")
    return files
