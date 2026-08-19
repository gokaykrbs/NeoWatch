"""
NeoWatch - NASA NeoWs API Client & Data Ingestion Engine
Handles rate limiting, 7-day chunk sliding windows, retries, and JSON flattening.
"""

import time
import json
import csv
import logging
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.config import (
    NASA_API_KEY,
    NASA_FEED_BASE_URL,
    REQUEST_DELAY_SECONDS,
    MAX_RETRIES,
    BACKOFF_FACTOR,
    RAW_DATA_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("NeoWatch.APIClient")


class NASAClient:
    """Client for querying and processing NASA's Near Earth Object Web Service (NeoWs) API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or NASA_API_KEY
        if not self.api_key or self.api_key == "DEMO_KEY":
            logger.warning("Using DEMO_KEY. Rate limit is strictly 30 requests/hour and 50 requests/day.")

    def fetch_feed_chunk(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Fetch NEO feed data for a maximum 7-day window.
        Dates must be in 'YYYY-MM-DD' format.
        """
        url = f"{NASA_FEED_BASE_URL}?start_date={start_date}&end_date={end_date}&api_key={self.api_key}"

        retries = 0
        current_delay = REQUEST_DELAY_SECONDS

        while retries <= MAX_RETRIES:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "NeoWatch/1.0"})
                with urllib.request.urlopen(req, timeout=25) as response:
                    if response.status == 200:
                        raw_data = response.read().decode("utf-8")
                        return json.loads(raw_data)

            except urllib.error.HTTPError as http_err:
                if http_err.code == 429:
                    logger.warning("HTTP 429 Too Many Requests. Backing off for %.1f seconds...", current_delay * 3)
                    time.sleep(current_delay * 3)
                    retries += 1
                    current_delay *= BACKOFF_FACTOR
                    continue
                else:
                    retries += 1
                    logger.error("HTTP Error %d (Attempt %d/%d): %s", http_err.code, retries, MAX_RETRIES, http_err.reason)
                    if retries > MAX_RETRIES:
                        raise
                    time.sleep(current_delay)
                    current_delay *= BACKOFF_FACTOR

            except Exception as exc:
                retries += 1
                logger.error("Request failed (Attempt %d/%d): %s", retries, MAX_RETRIES, exc)
                if retries > MAX_RETRIES:
                    raise
                time.sleep(current_delay)
                current_delay *= BACKOFF_FACTOR

        raise RuntimeError(f"Failed to fetch data from {start_date} to {end_date} after {MAX_RETRIES} retries.")

    def parse_feed_json(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten nested NeoWs feed JSON response into a tabular list of dictionaries.
        """
        records: List[Dict[str, Any]] = []
        near_earth_objects = data.get("near_earth_objects", {})

        for date_str, asteroids in near_earth_objects.items():
            for ast in asteroids:
                try:
                    # Basic attributes
                    ast_id = ast.get("id")
                    name = ast.get("name")
                    neo_reference_id = ast.get("neo_reference_id")
                    absolute_magnitude_h = ast.get("absolute_magnitude_h")
                    is_potentially_hazardous = ast.get("is_potentially_hazardous_asteroid", False)
                    is_sentry_object = ast.get("is_sentry_object", False)

                    # Diameter estimates in kilometers
                    est_diam = ast.get("estimated_diameter", {}).get("kilometers", {})
                    diam_min = est_diam.get("estimated_diameter_min")
                    diam_max = est_diam.get("estimated_diameter_max")
                    diam_mean = (diam_min + diam_max) / 2.0 if (diam_min is not None and diam_max is not None) else None

                    # Diameter in meters
                    est_diam_m = ast.get("estimated_diameter", {}).get("meters", {})
                    diam_min_m = est_diam_m.get("estimated_diameter_min")
                    diam_max_m = est_diam_m.get("estimated_diameter_max")

                    # Close approach details (take the matching close approach or first available)
                    close_approaches = ast.get("close_approach_data", [])
                    close_approach = None
                    if close_approaches:
                        for ca in close_approaches:
                            if ca.get("close_approach_date") == date_str:
                                close_approach = ca
                                break
                        if close_approach is None:
                            close_approach = close_approaches[0]

                    relative_vel_km_s = None
                    relative_vel_km_h = None
                    miss_dist_km = None
                    miss_dist_astronomical = None
                    miss_dist_lunar = None
                    orbiting_body = None
                    close_approach_date = date_str

                    if close_approach:
                        close_approach_date = close_approach.get("close_approach_date", date_str)
                        orbiting_body = close_approach.get("orbiting_body", "Earth")

                        rel_vel = close_approach.get("relative_velocity", {})
                        relative_vel_km_s = float(rel_vel["kilometers_per_second"]) if "kilometers_per_second" in rel_vel and rel_vel["kilometers_per_second"] is not None else None
                        relative_vel_km_h = float(rel_vel["kilometers_per_hour"]) if "kilometers_per_hour" in rel_vel and rel_vel["kilometers_per_hour"] is not None else None

                        miss_dist = close_approach.get("miss_distance", {})
                        miss_dist_km = float(miss_dist["kilometers"]) if "kilometers" in miss_dist and miss_dist["kilometers"] is not None else None
                        miss_dist_astronomical = float(miss_dist["astronomical"]) if "astronomical" in miss_dist and miss_dist["astronomical"] is not None else None
                        miss_dist_lunar = float(miss_dist["lunar"]) if "lunar" in miss_dist and miss_dist["lunar"] is not None else None

                    record = {
                        "id": ast_id,
                        "neo_reference_id": neo_reference_id,
                        "name": name,
                        "absolute_magnitude_h": float(absolute_magnitude_h) if absolute_magnitude_h is not None else None,
                        "estimated_diameter_min_km": float(diam_min) if diam_min is not None else None,
                        "estimated_diameter_max_km": float(diam_max) if diam_max is not None else None,
                        "estimated_diameter_mean_km": float(diam_mean) if diam_mean is not None else None,
                        "estimated_diameter_min_m": float(diam_min_m) if diam_min_m is not None else None,
                        "estimated_diameter_max_m": float(diam_max_m) if diam_max_m is not None else None,
                        "relative_velocity_km_s": relative_vel_km_s,
                        "relative_velocity_km_h": relative_vel_km_h,
                        "miss_distance_km": miss_dist_km,
                        "miss_distance_astronomical": miss_dist_astronomical,
                        "miss_distance_lunar": miss_dist_lunar,
                        "orbiting_body": orbiting_body,
                        "close_approach_date": close_approach_date,
                        "is_sentry_object": bool(is_sentry_object),
                        "is_potentially_hazardous_asteroid": int(bool(is_potentially_hazardous)),
                    }
                    records.append(record)

                except Exception as parse_err:
                    logger.error("Error parsing asteroid record %s: %s", ast.get("id"), parse_err)
                    continue

        return records

    def fetch_date_range(self, start_date_str: str, end_date_str: str, delay_between_calls: float = REQUEST_DELAY_SECONDS) -> List[Dict[str, Any]]:
        """
        Iteratively fetch NEO data across arbitrary date ranges by slicing into 7-day chunks.
        Returns a list of deduplicated record dictionaries.
        """
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        if start_date > end_date:
            raise ValueError(f"start_date ({start_date_str}) cannot be later than end_date ({end_date_str})")

        all_records: List[Dict[str, Any]] = []
        seen_keys = set()
        current_start = start_date

        total_days = (end_date - start_date).days + 1
        logger.info("Starting historical ingestion from %s to %s (%d days total)...", start_date_str, end_date_str, total_days)

        chunk_count = 0
        while current_start <= end_date:
            chunk_count += 1
            current_end = min(current_start + timedelta(days=6), end_date)

            s_str = current_start.strftime("%Y-%m-%d")
            e_str = current_end.strftime("%Y-%m-%d")

            logger.info("[Chunk %d] Requesting NEO feed: %s -> %s", chunk_count, s_str, e_str)
            try:
                json_data = self.fetch_feed_chunk(s_str, e_str)
                chunk_records = self.parse_feed_json(json_data)
                
                # Deduplicate records by (id, close_approach_date)
                for rec in chunk_records:
                    key = (rec.get("id"), rec.get("close_approach_date"))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_records.append(rec)

                logger.info("[Chunk %d] Fetched %d records. (Unique total: %d)", chunk_count, len(chunk_records), len(all_records))
            except Exception as err:
                logger.error("[Chunk %d] Error fetching chunk %s to %s: %s", chunk_count, s_str, e_str, err)

            current_start = current_end + timedelta(days=1)
            time.sleep(delay_between_calls)

        logger.info("Ingestion complete. Total unique records: %d", len(all_records))
        return all_records

    def save_to_csv(self, records: Any, output_path: Any = RAW_DATA_PATH) -> None:
        """Save records (list of dicts or DataFrame) to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_PANDAS and isinstance(records, pd.DataFrame):
            records.to_csv(output_path, index=False)
            logger.info("Saved %d records to %s using pandas.", len(records), output_path)
        elif isinstance(records, list) and records:
            fieldnames = list(records[0].keys())
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
            logger.info("Saved %d records to %s using csv.DictWriter.", len(records), output_path)
        elif HAS_PANDAS and isinstance(records, list) and records:
            df = pd.DataFrame(records)
            df.to_csv(output_path, index=False)
            logger.info("Saved %d records to %s using pandas.", len(df), output_path)
        else:
            logger.warning("No records to save.")

