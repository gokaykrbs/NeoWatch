"""
NeoWatch - Historical Data Collection Script
Executes batch extraction from NASA NeoWs API and outputs raw dataset to data/raw_asteroid_data.csv.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if "." not in sys.path:
    sys.path.insert(0, ".")

from src.api_client import NASAClient
from src.config import RAW_DATA_PATH, NASA_API_KEY


def main():
    parser = argparse.ArgumentParser(description="Collect historical asteroid data from NASA NeoWs API.")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date in YYYY-MM-DD format (default: 2023-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-01-01",
        help="End date in YYYY-MM-DD format (default: 2024-01-01)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RAW_DATA_PATH),
        help=f"Output CSV path (default: {RAW_DATA_PATH})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Delay in seconds between 7-day chunk API requests to respect rate limits",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NeoWatch - NASA Asteroid Data Ingestion Pipeline")
    print("=" * 70)
    print(f"Start Date : {args.start_date}")
    print(f"End Date   : {args.end_date}")
    print(f"Target CSV : {args.output}")
    print(f"API Key    : {'Configured (' + NASA_API_KEY[:6] + '...)' if NASA_API_KEY else 'Missing'}")
    print("=" * 70)

    client = NASAClient()
    try:
        records = client.fetch_date_range(args.start_date, args.end_date, delay_between_calls=args.delay)
        if not records:
            print("[ERROR] No data returned. Please check date range or API key.")
            sys.exit(1)

        client.save_to_csv(records, output_path=args.output)
        
        haz_count = sum(1 for r in records if r.get("is_potentially_hazardous_asteroid") == 1)
        safe_count = len(records) - haz_count
        haz_pct = (haz_count / len(records)) * 100 if records else 0.0

        print("\n" + "=" * 70)
        print("[SUCCESS] Data Collection Complete!")
        print(f"Total Records Collected : {len(records)}")
        print(f"Hazardous Asteroids     : {haz_count} ({haz_pct:.2f}%)")
        print(f"Non-Hazardous Asteroids : {safe_count}")
        print(f"File Saved At           : {args.output}")
        print("=" * 70)

    except Exception as err:
        print(f"[ERROR] Ingestion Error: {err}", file=sys.stderr)
        sys.exit(1)




if __name__ == "__main__":
    main()
