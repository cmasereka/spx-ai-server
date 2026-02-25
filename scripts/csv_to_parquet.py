#!/usr/bin/env python3
"""
Convert ThetaData CSV tick files → 1-minute parquet files.

Source layout (any depth under SOURCE_ROOT):
  theta_tick_WeekN_YYYYMMDD-YYYYMMDD/
    SPX_index_price_tick_YYYYMMDD.csv
    SPXW_option_quotes_tick_YYYYMMDD_expYYYYMMDD_sr200.csv

Target layout (matches ParquetDataLoader exactly):
  data/processed/parquet_1m/
    SPX_index_price_1m_YYYYMMDD.parquet
    SPXW_option_quotes_1m_YYYYMMDD_expYYYYMMDD_sr200.parquet

CSV  → Parquet column mapping
  SPX:
    timestamp  str        → timestamp  object (kept as-is, e.g. "2025-01-02T09:31:00")
    price      float64    → price      float32

  Options:
    symbol        str     → symbol        category
    expiration    str     → expiration    category  ("2025-01-02" format)
    strike        float64 → strike        float32
    right         str     → right         category  (CALL/PUT — kept as-is)
    timestamp     str     → timestamp     category
    bid_size      int64   → bid_size      uint16
    bid_exchange  int64   → bid_exchange  category  (int cast to str, e.g. "5")
    bid           float64 → bid           float32
    bid_condition int64   → bid_condition int64
    ask_size      int64   → ask_size      uint16
    ask_exchange  int64   → ask_exchange  category
    ask           float64 → ask           float32
    ask_condition int64   → ask_condition int64

Usage:
    # Convert everything, skip dates already done
    python scripts/csv_to_parquet.py --skip-existing

    # Force re-convert everything
    python scripts/csv_to_parquet.py

    # Preview without writing
    python scripts/csv_to_parquet.py --dry-run

    # Override source / target dirs
    python scripts/csv_to_parquet.py \\
        --source-dir /path/to/csv/root \\
        --output-dir /path/to/parquet_1m
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent
SOURCE_ROOT = Path("/Users/cmasereka/Personal/Trading/data")
OUTPUT_DIR  = REPO_ROOT / "data" / "processed" / "parquet_1m"


# ---------------------------------------------------------------------------
# CSV → Parquet converters
# ---------------------------------------------------------------------------

def convert_spx(csv_path: Path, out_path: Path) -> int:
    """
    Convert one SPX index CSV to parquet.
    Returns number of rows written.
    """
    df = pd.read_csv(csv_path, dtype={"timestamp": str, "price": "float32"})

    # Ensure column names match (defensive)
    df.columns = [c.strip().lower() for c in df.columns]
    if "price" not in df.columns or "timestamp" not in df.columns:
        raise ValueError(f"Unexpected SPX columns: {list(df.columns)}")

    df["price"] = df["price"].astype("float32")
    # timestamp stays as object/str — matches existing parquet
    df.to_parquet(out_path, index=False)
    return len(df)


def convert_options(csv_path: Path, out_path: Path) -> int:
    """
    Convert one SPXW options CSV to parquet.
    Returns number of rows written.
    """
    df = pd.read_csv(
        csv_path,
        dtype={
            "symbol":        str,
            "expiration":    str,
            "strike":        "float32",
            "right":         str,
            "timestamp":     str,
            "bid_size":      "int32",
            "bid_exchange":  str,
            "bid":           "float32",
            "bid_condition": "int64",
            "ask_size":      "int32",
            "ask_exchange":  str,
            "ask":           "float32",
            "ask_condition": "int64",
        },
    )

    df.columns = [c.strip().lower() for c in df.columns]

    # Strip stray quotes that some CSV writers leave around string fields
    for col in ("symbol", "expiration", "right"):
        df[col] = df[col].str.strip('"').str.strip()

    # bid_exchange / ask_exchange come in as integers in the CSV (e.g. 5)
    # but are stored as the string "5" in the existing parquet files.
    df["bid_exchange"] = df["bid_exchange"].astype(str).str.strip('"')
    df["ask_exchange"] = df["ask_exchange"].astype(str).str.strip('"')

    # Cast to final dtypes matching the existing parquet schema
    df["symbol"]        = df["symbol"].astype("category")
    df["expiration"]    = df["expiration"].astype("category")
    df["strike"]        = df["strike"].astype("float32")
    df["right"]         = df["right"].astype("category")
    df["timestamp"]     = df["timestamp"].astype("category")
    df["bid_size"]      = df["bid_size"].clip(0, 65535).astype("uint16")
    df["bid_exchange"]  = df["bid_exchange"].astype("category")
    df["bid"]           = df["bid"].astype("float32")
    df["bid_condition"] = df["bid_condition"].astype("int64")
    df["ask_size"]      = df["ask_size"].clip(0, 65535).astype("uint16")
    df["ask_exchange"]  = df["ask_exchange"].astype("category")
    df["ask"]           = df["ask"].astype("float32")
    df["ask_condition"] = df["ask_condition"].astype("int64")

    df.to_parquet(out_path, index=False)
    return len(df)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _out_name_spx(csv_name: str) -> str:
    """SPX_index_price_tick_YYYYMMDD.csv → SPX_index_price_1m_YYYYMMDD.parquet"""
    return csv_name.replace("_tick_", "_1m_").replace(".csv", ".parquet")


def _out_name_opt(csv_name: str) -> str:
    """SPXW_option_quotes_tick_YYYYMMDD_expYYYYMMDD_sr200.csv
       → SPXW_option_quotes_1m_YYYYMMDD_expYYYYMMDD_sr200.parquet"""
    return csv_name.replace("_tick_", "_1m_").replace(".csv", ".parquet")


def discover_pairs(source_root: Path):
    """
    Walk source_root recursively and yield (spx_csv, opt_csv, date_str) tuples.
    A day is only yielded when both the SPX and options CSV exist.
    """
    spx_files = {f.stem.split("tick_")[-1]: f
                 for f in source_root.rglob("SPX_index_price_tick_*.csv")}
    opt_files = {f.stem.split("tick_")[-1].split("_exp")[0]: f
                 for f in source_root.rglob("SPXW_option_quotes_tick_*.csv")}

    matched = sorted(set(spx_files) & set(opt_files))
    missing_opt = set(spx_files) - set(opt_files)
    missing_spx = set(opt_files) - set(spx_files)

    if missing_opt:
        logger.warning(f"{len(missing_opt)} SPX file(s) have no matching options CSV: "
                       f"{sorted(missing_opt)[:5]}{'...' if len(missing_opt) > 5 else ''}")
    if missing_spx:
        logger.warning(f"{len(missing_spx)} options file(s) have no matching SPX CSV: "
                       f"{sorted(missing_spx)[:5]}{'...' if len(missing_spx) > 5 else ''}")

    for date_str in matched:
        yield spx_files[date_str], opt_files[date_str], date_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert ThetaData CSV tick files to 1-minute parquet"
    )
    parser.add_argument(
        "--source-dir", default=str(SOURCE_ROOT),
        help=f"Root folder containing week subfolders with CSVs (default: {SOURCE_ROOT})"
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help=f"Destination parquet folder (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip dates where both parquet files already exist in output-dir"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be converted without writing anything"
    )
    args = parser.parse_args()

    source_root = Path(args.source_dir)
    output_dir  = Path(args.output_dir)

    if not source_root.exists():
        logger.error(f"Source directory not found: {source_root}")
        sys.exit(1)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all matching CSV pairs
    pairs = list(discover_pairs(source_root))
    if not pairs:
        logger.error(f"No matching CSV pairs found under {source_root}")
        sys.exit(1)

    logger.info(f"Found {len(pairs)} trading days to convert "
                f"({pairs[0][2]} → {pairs[-1][2]})")

    ok = skipped = failed = 0

    for i, (spx_csv, opt_csv, date_str) in enumerate(pairs, 1):
        spx_out = output_dir / _out_name_spx(spx_csv.name)
        opt_out = output_dir / _out_name_opt(opt_csv.name)

        if args.skip_existing and spx_out.exists() and opt_out.exists():
            logger.debug(f"[{i}/{len(pairs)}] {date_str}: already exists, skipping")
            skipped += 1
            continue

        if args.dry_run:
            logger.info(f"[{i}/{len(pairs)}] {date_str}: [dry-run]"
                        f"\n    {spx_csv.name} → {spx_out.name}"
                        f"\n    {opt_csv.name} → {opt_out.name}")
            ok += 1
            continue

        try:
            spx_rows = convert_spx(spx_csv, spx_out)
            opt_rows = convert_options(opt_csv, opt_out)
            logger.info(f"[{i}/{len(pairs)}] {date_str}: "
                        f"SPX {spx_rows} rows, options {opt_rows:,} rows")
            ok += 1
        except Exception as e:
            logger.error(f"[{i}/{len(pairs)}] {date_str}: FAILED — {e}")
            # Remove partial output so the day isn't half-converted
            spx_out.unlink(missing_ok=True)
            opt_out.unlink(missing_ok=True)
            failed += 1

    # Summary
    total = ok + skipped + failed
    logger.info(
        f"\nDone. {ok} converted, {skipped} skipped, {failed} failed "
        f"out of {total} days."
    )
    if not args.dry_run:
        logger.info(f"Parquet files written to: {output_dir}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
    main()
