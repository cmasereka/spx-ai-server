"""
TastyTrade live data smoke test.

Connects to TastyTrade, fetches the real SPX price via the market-data REST
endpoint every 5 seconds, and prints spread pricing for:
  - Put spread  : short = SPX - 30, long = SPX - 40  (width $10)
  - Call spread : short = SPX + 30, long = SPX + 40  (width $10)

Option bid/ask/delta come from DXLink streaming (live quotes and greeks).
The SPX price is fetched directly from the REST API — no estimation.

Run:
    python test_tastytrade_live.py [--iterations N]   (default 10)

Credentials are read from .env:
    TASTYTRADE_PROVIDER_SECRET
    TASTYTRADE_REFRESH_TOKEN
"""

import argparse
import asyncio
import os
import sys
from collections import defaultdict
from datetime import datetime, date, timedelta

from dotenv import load_dotenv

load_dotenv()

PROVIDER_SECRET = os.getenv("TASTYTRADE_PROVIDER_SECRET", "")
REFRESH_TOKEN   = os.getenv("TASTYTRADE_REFRESH_TOKEN", "")

if not PROVIDER_SECRET or not REFRESH_TOKEN:
    print("ERROR: TASTYTRADE_PROVIDER_SECRET and TASTYTRADE_REFRESH_TOKEN must be set in .env")
    sys.exit(1)

try:
    from tastytrade import Session
    from tastytrade.instruments import NestedOptionChain
    from tastytrade.market_data import get_market_data_by_type
    from tastytrade.streamer import DXLinkStreamer
    from tastytrade.dxfeed import Quote, Greeks
except ImportError:
    print("ERROR: tastytrade package not found — run: pip install tastytrade")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Shared state (written by stream listeners, read by poll loop)
# ---------------------------------------------------------------------------
quotes:      dict[str, dict] = defaultdict(dict)  # sym → {bid, ask}
greeks_data: dict[str, dict] = defaultdict(dict)  # sym → {delta}
symbol_meta: dict[str, dict] = {}                 # sym → {strike, option_type}

SPREAD_DISTANCE = 30   # short strike is this far OTM from SPX
SPREAD_WIDTH    = 10   # long strike is this much further OTM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_to_nearest(value: float, step: float = 5.0) -> float:
    """Round to the nearest available SPX strike increment ($5)."""
    return round(round(value / step) * step, 2)


def _find_symbol(strike: float, option_type: str) -> str | None:
    for sym, meta in symbol_meta.items():
        if meta["option_type"] == option_type and meta["strike"] == strike:
            return sym
    return None


def _spread_info(short_strike: float, long_strike: float, option_type: str) -> dict:
    short_sym = _find_symbol(short_strike, option_type)
    long_sym  = _find_symbol(long_strike,  option_type)

    short_bid   = quotes.get(short_sym, {}).get("bid", None) if short_sym else None
    short_ask   = quotes.get(short_sym, {}).get("ask", None) if short_sym else None
    long_bid    = quotes.get(long_sym,  {}).get("bid", None) if long_sym  else None
    long_ask    = quotes.get(long_sym,  {}).get("ask", None) if long_sym  else None
    short_delta = greeks_data.get(short_sym, {}).get("delta", None) if short_sym else None
    long_delta  = greeks_data.get(long_sym,  {}).get("delta", None) if long_sym  else None

    net_credit: float | None = None
    if short_bid is not None and long_ask is not None:
        net_credit = round(float(short_bid) - float(long_ask), 2)

    return {
        "short_strike": short_strike, "long_strike":  long_strike,
        "short_bid": short_bid, "short_ask": short_ask, "short_delta": short_delta,
        "long_bid":  long_bid,  "long_ask":  long_ask,  "long_delta":  long_delta,
        "net_credit": net_credit,
    }


def _log_spread(label: str, info: dict):
    def _fmt(v, decimals=2):
        return f"{float(v):.{decimals}f}" if v is not None else "N/A"

    credit = f"${info['net_credit']:.2f}" if info["net_credit"] is not None else "N/A"
    print(f"  {label}")
    print(f"    Short {info['short_strike']:.0f}  "
          f"bid={_fmt(info['short_bid'])}  ask={_fmt(info['short_ask'])}  "
          f"delta={_fmt(info['short_delta'], 3)}")
    print(f"    Long  {info['long_strike']:.0f}  "
          f"bid={_fmt(info['long_bid'])}  ask={_fmt(info['long_ask'])}  "
          f"delta={_fmt(info['long_delta'], 3)}")
    print(f"    Net credit: {credit}")


# ---------------------------------------------------------------------------
# Stream listeners (option quotes and greeks only)
# ---------------------------------------------------------------------------

async def listen_quotes(streamer: DXLinkStreamer):
    async for event in streamer.listen(Quote):
        sym = getattr(event, "event_symbol", None)
        if sym and sym in symbol_meta:
            quotes[sym]["bid"] = event.bid_price
            quotes[sym]["ask"] = event.ask_price


async def listen_greeks(streamer: DXLinkStreamer):
    async for event in streamer.listen(Greeks):
        sym = getattr(event, "event_symbol", None)
        if sym and sym in symbol_meta:
            greeks_data[sym]["delta"] = event.delta


# ---------------------------------------------------------------------------
# Poll loop — REST call for SPX price each iteration
# ---------------------------------------------------------------------------

async def poll_loop(session: Session, iterations: int, interval: float = 5.0):
    for i in range(1, iterations + 1):
        await asyncio.sleep(interval)
        now = datetime.now().strftime("%H:%M:%S")

        # Fetch live SPX price directly from TastyTrade REST API
        # REST API uses "SPX" (not the DXFeed streaming symbol "$SPX.X")
        try:
            results = await get_market_data_by_type(session, indices=["SPX"])
            spx_data = results[0] if results else None
            spx = float(spx_data.mark) if spx_data and float(spx_data.mark) > 0 else 0.0
        except Exception as e:
            print(f"[{now}]  Iteration {i}/{iterations}  ERROR fetching SPX: {e}")
            continue

        if spx <= 0:
            print(f"[{now}]  Iteration {i}/{iterations}  SPX price unavailable")
            continue

        print(f"\n{'='*60}")
        print(f"[{now}]  Iteration {i}/{iterations}  SPX = {spx:.2f}")
        print(f"{'='*60}")

        put_short = _round_to_nearest(spx - SPREAD_DISTANCE)
        put_long  = _round_to_nearest(spx - SPREAD_DISTANCE - SPREAD_WIDTH)
        _log_spread(f"PUT  SPREAD  (short {put_short:.0f} / long {put_long:.0f})",
                    _spread_info(put_short, put_long, "put"))
        print()
        call_short = _round_to_nearest(spx + SPREAD_DISTANCE)
        call_long  = _round_to_nearest(spx + SPREAD_DISTANCE + SPREAD_WIDTH)
        _log_spread(f"CALL SPREAD  (short {call_short:.0f} / long {call_long:.0f})",
                    _spread_info(call_short, call_long, "call"))

    print("\nDone.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(iterations: int):
    print("Connecting to TastyTrade (production)...")
    session = Session(PROVIDER_SECRET, REFRESH_TOKEN, is_test=False)
    print("Session OK")

    today = date.today()
    print(f"Loading SPXW 0DTE option chain for {today}...")
    chains = await NestedOptionChain.get(session, "SPXW")

    today_expir = None
    for chain in chains:
        for expir in chain.expirations:
            if expir.expiration_date == today:
                today_expir = expir
                break
        if today_expir:
            break

    if today_expir is None:
        print(f"ERROR: No SPXW 0DTE options found for {today}. Market may be closed.")
        return

    for strike_entry in today_expir.strikes:
        strike = float(strike_entry.strike_price)
        if strike_entry.call_streamer_symbol:
            symbol_meta[strike_entry.call_streamer_symbol] = {"strike": strike, "option_type": "call"}
        if strike_entry.put_streamer_symbol:
            symbol_meta[strike_entry.put_streamer_symbol] = {"strike": strike, "option_type": "put"}

    print(f"Loaded {len(symbol_meta)} option symbols")

    option_symbols = list(symbol_meta.keys())

    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(Quote, option_symbols)
        await streamer.subscribe(Greeks, option_symbols)
        print(f"Subscribed to {len(option_symbols)} option symbols (quotes + greeks)")
        print(f"Starting {iterations} iterations every 5 seconds...\n")

        async with asyncio.TaskGroup() as tg:
            tg.create_task(listen_quotes(streamer))
            tg.create_task(listen_greeks(streamer))
            tg.create_task(poll_loop(session, iterations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TastyTrade live data smoke test")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of 5-second polling iterations (default: 10)")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.iterations))
    except KeyboardInterrupt:
        print("\nInterrupted.")
