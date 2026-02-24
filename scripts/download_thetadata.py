#!/usr/bin/env python3
"""
ThetaData 1-minute SPX + SPXW options downloader.

Downloads data directly from the ThetaData Terminal REST API
(must be running on localhost:25503) and saves to the parquet layout
that ParquetDataLoader expects:

  data/processed/parquet_1m/
    SPX_index_price_1m_YYYYMMDD.parquet
    SPXW_option_quotes_1m_YYYYMMDD_expYYYYMMDD_sr200.parquet

Usage:
    # Last 30 trading days
    python scripts/download_thetadata.py --days-back 30

    # Specific date range
    python scripts/download_thetadata.py --start-date 2025-01-02 --end-date 2025-12-31

    # Single date
    python scripts/download_thetadata.py --date 2025-06-10

    # Dry-run: show what would be downloaded without saving
    python scripts/download_thetadata.py --start-date 2025-01-02 --end-date 2025-01-10 --dry-run

    # Skip dates already on disk
    python scripts/download_thetadata.py --days-back 90 --skip-existing
"""

import argparse
import sys
import os
import time
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import Optional, List

import requests
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "parquet_1m"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ThetaData Terminal config
# ---------------------------------------------------------------------------
TERMINAL_URL  = "http://127.0.0.1:25503"
STRIKE_RADIUS = 200   # Download strikes within ±$200 of SPX at open
REQUEST_DELAY = 0.25  # Seconds between API calls (be polite to Terminal)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(endpoint: str, params: dict, retries: int = 3) -> Optional[dict]:
    """GET from ThetaData Terminal with retry logic."""
    url = f"{TERMINAL_URL}{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP {r.status_code} on {url} {params}: {e}")
            return None
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt+1}/{retries} failed ({e}), retrying in {wait}s")
            time.sleep(wait)
    return None


def _trading_days(start: date, end: date) -> List[date]:
    """Return weekdays in [start, end]. Does not filter US holidays."""
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # Mon–Fri
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _fmt(d: date) -> str:
    """YYYYMMDD format for ThetaData params."""
    return d.strftime("%Y%m%d")


def _spx_file(d: date) -> Path:
    return OUTPUT_DIR / f"SPX_index_price_1m_{_fmt(d)}.parquet"


def _opt_file(d: date) -> Path:
    return OUTPUT_DIR / f"SPXW_option_quotes_1m_{_fmt(d)}_exp{_fmt(d)}_sr200.parquet"


def check_terminal() -> bool:
    """Verify Terminal is reachable."""
    try:
        r = requests.get(f"{TERMINAL_URL}/v2/system/mdds/status", timeout=5)
        return r.status_code < 500
    except Exception:
        return False

# ---------------------------------------------------------------------------
# SPX 1-minute index price
# ---------------------------------------------------------------------------

def download_spx(d: date) -> Optional[pd.DataFrame]:
    """
    Download 1-minute SPX index prices for a single trading day.
    Returns DataFrame with columns: timestamp (str), price (float32).
    """
    data = _get("/v2/hist/index/quotes", {
        "root":       "SPX",
        "start_date": _fmt(d),
        "end_date":   _fmt(d),
        "ivl":        60000,    # 60 000 ms = 1 minute
    })
    time.sleep(REQUEST_DELAY)

    if not data or "response" not in data:
        logger.error(f"No SPX data returned for {d}")
        return None

    rows = []
    for entry in data["response"]:
        # ThetaData returns ms-since-epoch for the bar open time
        ms   = entry.get("ms_of_day", 0)
        price = entry.get("close", entry.get("mid", 0.0))
        # Convert ms-of-day to HH:MM:SS
        total_sec = ms // 1000
        h, rem = divmod(total_sec, 3600)
        m, s   = divmod(rem, 60)
        ts = f"{d.isoformat()}T{h:02d}:{m:02d}:{s:02d}"
        rows.append({"timestamp": ts, "price": float(price)})

    if not rows:
        logger.error(f"Empty SPX response for {d}")
        return None

    df = pd.DataFrame(rows)
    df["price"] = df["price"].astype("float32")
    return df

# ---------------------------------------------------------------------------
# SPXW 0DTE option quotes — 1-minute bid/ask
# ---------------------------------------------------------------------------

def _get_spx_open(d: date, spx_df: pd.DataFrame) -> float:
    """Return the first non-zero price of the day as the 'open'."""
    nonzero = spx_df.loc[spx_df["price"] > 0, "price"]
    return float(nonzero.iloc[0]) if len(nonzero) else 5000.0


def download_options(d: date, spx_open: float) -> Optional[pd.DataFrame]:
    """
    Download 1-minute SPXW 0DTE option quotes.
    Iterates over all strikes within ±STRIKE_RADIUS of spx_open, both P and C.
    Returns DataFrame matching the existing options parquet schema.
    """
    strike_lo = int(round(spx_open / 5) * 5) - STRIKE_RADIUS
    strike_hi = int(round(spx_open / 5) * 5) + STRIKE_RADIUS
    strikes   = range(strike_lo, strike_hi + 1, 5)

    all_rows: list[dict] = []

    for right in ("C", "P"):
        for strike in strikes:
            data = _get("/v2/hist/option/quotes", {
                "root":       "SPXW",
                "exp":        _fmt(d),
                "strike":     strike * 1000,   # ThetaData uses strike × 1000
                "right":      right,
                "start_date": _fmt(d),
                "end_date":   _fmt(d),
                "ivl":        60000,
            })
            time.sleep(REQUEST_DELAY)

            if not data or "response" not in data:
                continue

            for entry in data["response"]:
                ms = entry.get("ms_of_day", 0)
                total_sec = ms // 1000
                h, rem = divmod(total_sec, 3600)
                m, s   = divmod(rem, 60)
                ts = f"{d.isoformat()}T{h:02d}:{m:02d}:{s:02d}"

                all_rows.append({
                    "symbol":        "SPXW",
                    "expiration":    d.isoformat(),
                    "strike":        float(strike),
                    "right":         right,
                    "timestamp":     ts,
                    "bid_size":      int(entry.get("bid_size",  0)),
                    "bid_exchange":  str(entry.get("bid_exchange", "")),
                    "bid":           float(entry.get("bid",  0.0)),
                    "bid_condition": int(entry.get("bid_condition", 0)),
                    "ask_size":      int(entry.get("ask_size",  0)),
                    "ask_exchange":  str(entry.get("ask_exchange", "")),
                    "ask":           float(entry.get("ask",  0.0)),
                    "ask_condition": int(entry.get("ask_condition", 0)),
                })

        logger.debug(f"  {d} {right}: {len(all_rows)} rows so far")

    if not all_rows:
        logger.error(f"No options data returned for {d}")
        return None

    df = pd.DataFrame(all_rows)

    # Match schema of existing parquet files
    df["symbol"]     = df["symbol"].astype("category")
    df["expiration"] = df["expiration"].astype("category")
    df["strike"]     = df["strike"].astype("float32")
    df["right"]      = df["right"].astype("category")
    df["timestamp"]  = df["timestamp"].astype("category")
    df["bid_size"]   = df["bid_size"].astype("uint16")
    df["bid_exchange"] = df["bid_exchange"].astype("category")
    df["bid"]        = df["bid"].astype("float32")
    df["ask_size"]   = df["ask_size"].astype("uint16")
    df["ask_exchange"] = df["ask_exchange"].astype("category")
    df["ask"]        = df["ask"].astype("float32")

    return df

# ---------------------------------------------------------------------------
# Per-day orchestration
# ---------------------------------------------------------------------------

def download_day(d: date, skip_existing: bool, dry_run: bool) -> bool:
    """Download SPX + options for one day. Returns True on success."""
    spx_path = _spx_file(d)
    opt_path = _opt_file(d)

    if skip_existing and spx_path.exists() and opt_path.exists():
        logger.info(f"{d}: already on disk, skipping")
        return True

    if dry_run:
        logger.info(f"{d}: [dry-run] would download SPX + SPXW options")
        return True

    logger.info(f"{d}: downloading SPX index prices …")
    spx_df = download_spx(d)
    if spx_df is None:
        return False

    spx_df.to_parquet(spx_path, index=False)
    logger.info(f"{d}: SPX saved ({len(spx_df)} rows) → {spx_path.name}")

    spx_open = _get_spx_open(d, spx_df)
    logger.info(f"{d}: downloading SPXW options (open≈{spx_open:.0f}, ±{STRIKE_RADIUS}pts) …")

    opt_df = download_options(d, spx_open)
    if opt_df is None:
        # Remove the SPX file so the day isn't half-downloaded
        spx_path.unlink(missing_ok=True)
        return False

    opt_df.to_parquet(opt_path, index=False)
    logger.info(f"{d}: options saved ({len(opt_df)} rows) → {opt_path.name}")
    return True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download 1-minute SPX + SPXW data from ThetaData Terminal"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date",       help="Single date  YYYY-MM-DD")
    group.add_argument("--days-back",  type=int, help="Trading days back from today")
    group.add_argument("--start-date", help="Range start  YYYY-MM-DD (requires --end-date)")

    parser.add_argument("--end-date",      default=None, help="Range end  YYYY-MM-DD")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip dates where both parquet files already exist")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print what would be downloaded without saving anything")
    parser.add_argument("--terminal-url",  default=TERMINAL_URL,
                        help=f"ThetaData Terminal base URL (default: {TERMINAL_URL})")

    args = parser.parse_args()

    # Override terminal URL if provided
    if args.terminal_url != TERMINAL_URL:
        # reassign via sys.modules so _get() picks it up
        sys.modules[__name__].TERMINAL_URL = args.terminal_url

    # Build date list
    today = date.today()
    if args.date:
        days = [date.fromisoformat(args.date)]
    elif args.days_back:
        end   = today - timedelta(days=1)
        start = end - timedelta(days=args.days_back * 2)  # overshoot, filter weekends
        days  = _trading_days(start, end)[-args.days_back:]
    else:
        if not args.end_date:
            parser.error("--start-date requires --end-date")
        days = _trading_days(
            date.fromisoformat(args.start_date),
            date.fromisoformat(args.end_date)
        )

    if not days:
        logger.error("No trading days in the requested range.")
        sys.exit(1)

    logger.info(f"{'[DRY-RUN] ' if args.dry_run else ''}Planning to download {len(days)} days "
                f"({days[0]} → {days[-1]})")

    if not args.dry_run:
        logger.info("Checking ThetaData Terminal connection …")
        if not check_terminal():
            logger.error(
                f"Cannot reach ThetaData Terminal at {TERMINAL_URL}.\n"
                "Make sure the Terminal app is running and logged in."
            )
            sys.exit(1)
        logger.info("Terminal reachable.")

    ok = fail = 0
    for i, d in enumerate(days, 1):
        logger.info(f"[{i}/{len(days)}] {d}")
        if download_day(d, args.skip_existing, args.dry_run):
            ok += 1
        else:
            fail += 1
            logger.warning(f"{d}: FAILED — will continue with remaining days")

    logger.info(f"Done. {ok} succeeded, {fail} failed out of {len(days)} days.")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    # Set up clean logging to console
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
    main()
