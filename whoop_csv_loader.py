"""
WHOOP CSV Loader
================
Reads whoop_data.csv and converts WHOOP HRV data into the same
dictionary structure returned by extract_hrv_features() in hrv_engine.py.

WHOOP exports RMSSD as "Heart rate variability (ms)".
The remaining metrics are estimated from RMSSD using validated approximations:
  - SDNN   = HRV * 1.2         (SDNN typically 20% above RMSSD in short-term)
  - pNN50  = max(0, (HRV - 20) / 1.5)  (linear fit to normative ranges)
  - SD2/SD1 = 2.0 - (HRV / 80)  (higher HRV → lower sympathetic ratio)

Output keys: rmssd, sdnn, pnn50, sd2_sd1  (sd1/sd2 omitted — not available)
"""

import csv
from pathlib import Path

CSV_PATH        = Path(__file__).parent / "whoop_data.csv"
SAMPLE_CSV_PATH = Path(__file__).parent / "sample_whoop_data.csv"

COL_HRV  = "Heart rate variability (ms)"
COL_RHR  = "Resting heart rate (bpm)"
COL_DATE = "Cycle start time"


def _estimate_features(rmssd: float) -> dict:
    """Derive pipeline-compatible HRV features from a single RMSSD value."""
    sdnn    = round(rmssd * 1.2, 2)
    pnn50   = round(max(0.0, (rmssd - 20.0) / 1.5), 2)
    sd2_sd1 = round(max(0.5, 2.0 - (rmssd / 80.0)), 3)
    return {
        "rmssd":   round(rmssd, 2),
        "sdnn":    sdnn,
        "pnn50":   pnn50,
        "sd2_sd1": sd2_sd1,
    }


def _load_rows(csv_path: Path = CSV_PATH) -> list[dict]:
    """Parse CSV and return cleaned rows with HRV present."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hrv_raw = row.get(COL_HRV, "").strip()
            if not hrv_raw:
                continue
            try:
                rmssd = float(hrv_raw)
            except ValueError:
                continue
            rhr_raw = row.get(COL_RHR, "").strip()
            rhr = float(rhr_raw) if rhr_raw else None
            rows.append({
                "date":  row.get(COL_DATE, "").strip(),
                "rmssd": rmssd,
                "rhr":   rhr,
            })
    return rows


def load_whoop_cycle(index: int) -> dict:
    """
    Return one day's HRV features as a dict compatible with hrv_engine output.

    index 0 = most recent cycle (top of CSV).
    Raises IndexError if index is out of range.
    """
    rows = _load_rows()
    if index < 0 or index >= len(rows):
        raise IndexError(f"index {index} out of range — {len(rows)} valid cycles available")
    row = rows[index]
    features = _estimate_features(row["rmssd"])
    features["date"] = row["date"]
    if row["rhr"] is not None:
        features["rhr_bpm"] = row["rhr"]
    return features


def load_all_whoop_cycles() -> list[dict]:
    """Return a list of feature dicts for all valid (non-empty HRV) cycles."""
    rows = _load_rows()
    cycles = []
    for row in rows:
        features = _estimate_features(row["rmssd"])
        features["date"] = row["date"]
        if row["rhr"] is not None:
            features["rhr_bpm"] = row["rhr"]
        cycles.append(features)
    return cycles


def load_whoop_weekly_windows(window_size: int = 7) -> list[dict]:
    """
    Group daily HRV data into non-overlapping windows of `window_size` days.

    Returns a list of feature dicts (oldest window first), each containing:
      - All keys from _estimate_features() derived from the window mean RMSSD
      - "date"       : "YYYY-MM-DD to YYYY-MM-DD"  (date range of the window)
      - "date_start" : first day in the window (YYYY-MM-DD)
      - "date_end"   : last day in the window (YYYY-MM-DD)
      - "day_count"  : number of days in this window (may be < window_size for last)
      - "hrv_values" : list of daily RMSSD values in the window
      - "mean_rmssd" : mean RMSSD across the window

    The last window may be smaller than window_size if total days is not divisible.
    Partial windows with at least 1 valid day are included.
    """
    rows = _load_rows()
    if not rows:
        return []

    # CSV is newest-first; reverse to process oldest → newest
    rows = list(reversed(rows))

    windows = []
    for i in range(0, len(rows), window_size):
        chunk = rows[i : i + window_size]
        rmssd_values = [r["rmssd"] for r in chunk]
        mean_rmssd   = sum(rmssd_values) / len(rmssd_values)

        dates = [r["date"][:10] for r in chunk if r["date"]]
        date_start = dates[0]  if dates else ""
        date_end   = dates[-1] if dates else ""

        features = _estimate_features(mean_rmssd)
        features["date"]       = f"{date_start} to {date_end}"
        features["date_start"] = date_start
        features["date_end"]   = date_end
        features["day_count"]  = len(chunk)
        features["hrv_values"] = rmssd_values
        features["mean_rmssd"] = round(mean_rmssd, 2)
        windows.append(features)

    return windows


def load_sample_weeks(window_size: int = 7) -> list[dict]:
    """
    Load the 3 bundled high-variance demo weeks from sample_whoop_data.csv.

    These three weeks were selected from the full WHOOP dataset for maximum
    HRV variability (std ≥ 14 ms), making the ISO arc most audibly dynamic:
      Week A: 2025-09-23 → 2025-09-29  std=15.9 ms
      Week B: 2025-10-01 → 2025-10-07  std=14.8 ms
      Week C: 2025-10-22 → 2025-10-28  std=21.0 ms  (min=56, max=122 ms)

    Returns the same format as load_whoop_weekly_windows().
    """
    if not SAMPLE_CSV_PATH.exists():
        return []
    rows = _load_rows(SAMPLE_CSV_PATH)
    if not rows:
        return []
    # CSV is newest-first; reverse to chronological order
    rows = list(reversed(rows))
    windows = []
    for i in range(0, len(rows), window_size):
        chunk = rows[i : i + window_size]
        rmssd_values = [r["rmssd"] for r in chunk]
        mean_rmssd   = sum(rmssd_values) / len(rmssd_values)
        dates = [r["date"][:10] for r in chunk if r["date"]]
        date_start = dates[0]  if dates else ""
        date_end   = dates[-1] if dates else ""
        features = _estimate_features(mean_rmssd)
        features["date"]       = f"{date_start} to {date_end}"
        features["date_start"] = date_start
        features["date_end"]   = date_end
        features["day_count"]  = len(chunk)
        features["hrv_values"] = rmssd_values
        features["mean_rmssd"] = round(mean_rmssd, 2)
        windows.append(features)
    return windows


def get_whoop_stats() -> None:
    """Print summary statistics for all valid WHOOP cycles."""
    rows = _load_rows()
    if not rows:
        print("No valid WHOOP cycles found.")
        return

    hrvs  = [r["rmssd"] for r in rows]
    dates = [r["date"]  for r in rows if r["date"]]

    total    = len(rows)
    mean_hrv = sum(hrvs) / total
    min_hrv  = min(hrvs)
    max_hrv  = max(hrvs)
    date_min = min(dates) if dates else "unknown"
    date_max = max(dates) if dates else "unknown"

    print(f"Total days   : {total}")
    print(f"Mean HRV     : {mean_hrv:.1f} ms")
    print(f"Min HRV      : {min_hrv:.1f} ms")
    print(f"Max HRV      : {max_hrv:.1f} ms")
    print(f"Date range   : {date_min}  to  {date_max}")


if __name__ == "__main__":
    print("=== WHOOP Stats ===")
    get_whoop_stats()

    print("\n=== Cycle 0 (most recent) ===")
    c = load_whoop_cycle(0)
    for k, v in c.items():
        print(f"  {k}: {v}")

    print(f"\n=== All cycles ({len(load_all_whoop_cycles())} total) ===")
    for i, cycle in enumerate(load_all_whoop_cycles()):
        print(f"  [{i}] {cycle['date'][:10]}  rmssd={cycle['rmssd']}  sdnn={cycle['sdnn']}  pnn50={cycle['pnn50']}  sd2_sd1={cycle['sd2_sd1']}")
