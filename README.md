# SPX AI Trading Platform

A production-grade Python backtesting platform for SPX (S&P 500 Index) 0DTE (same-day expiration) options strategies. Built around a FastAPI server with real-time WebSocket updates, PostgreSQL persistence, and a sophisticated intraday scan engine that scans every 1-minute bar from market open to 2:00 PM ET.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Server                          │
│  REST API (CRUD)  +  WebSocket (real-time progress feed)   │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   BacktestService   │  (api/backtest_service.py)
              │  async orchestration│
              └──────────┬──────────┘
                         │
     ┌───────────────────▼───────────────────┐
     │      EnhancedMultiStrategyEngine       │  (enhanced_multi_strategy.py)
     │  Intraday scan loop  09:35 → 14:00    │
     │  Risk guards  ·  Strike selection     │
     └────────────┬──────────────────────────┘
                  │
     ┌────────────▼────────────┐   ┌──────────────────────────┐
     │  IntradayPositionMonitor│   │  TechnicalAnalyzer       │
     │  (delta_strike_selector)│   │  RSI · VWAP · Bollinger  │
     └────────────┬────────────┘   └──────────────────────────┘
                  │
     ┌────────────▼────────────┐   ┌──────────────────────────┐
     │  ParquetDataLoader      │   │  PostgreSQL (via         │
     │  data/processed/        │   │  SQLAlchemy + Alembic)   │
     │  parquet_1m/            │   └──────────────────────────┘
     └─────────────────────────┘
```

## Project Structure

```
spx-ai-server/
├── main.py                       # FastAPI entry point, all routes
├── enhanced_multi_strategy.py    # Core intraday scan engine
├── enhanced_backtest.py          # Backtest result dataclasses
├── delta_strike_selector.py      # Strike selection + position monitor
│
├── api/
│   ├── models.py                 # Pydantic request/response models
│   ├── backtest_service.py       # Async orchestration + DB persistence
│   ├── database_routes.py        # Database query endpoints
│   └── websocket_manager.py      # WebSocket connection manager
│
├── src/
│   ├── data/
│   │   ├── parquet_loader.py     # Parquet file reader (SPX + options)
│   │   └── query_engine.py       # Optimised option chain queries
│   ├── database/
│   │   ├── connection.py         # SQLAlchemy engine + session factory
│   │   └── models.py             # ORM table definitions
│   └── strategies/
│
├── scripts/
│   ├── csv_to_parquet.py         # Convert ThetaData CSV → Parquet
│   └── download_thetadata.py     # Download data from ThetaData Terminal
│
├── tests/
│   ├── unit/
│   │   ├── test_delta_strike_selector.py
│   │   └── test_technical_analysis.py
│   ├── integration/
│   └── performance/
│
├── alembic/                      # Database migrations
│   └── versions/
│
├── config/settings.py            # Environment variable loading
├── .env.template                 # Environment variable template
├── requirements.txt
├── alembic.ini
└── start_api.sh                  # Convenience server start script
```

## Setup & Installation

### Prerequisites

- Python 3.8+
- PostgreSQL 13+
- ThetaData Terminal running locally (for live data or new downloads)
- Parquet data files in `data/processed/parquet_1m/`

### Installation

```bash
git clone <repo-url>
cd spx-ai-server

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Copy `.env.template` to `.env` and fill in your values:

```env
# ThetaData credentials (only needed for downloading new data)
THETA_USERNAME=your@email.com
THETA_PASSWORD=yourpassword

# Optional overrides
THETA_TERMINAL_PORT=25503     # default: 25503
SPX_SYMBOL=SPXW               # default: SPXW

# PostgreSQL connection
DATABASE_URL=postgresql://localhost:5432/spx_ai?gssencmode=disable
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=spx_ai
POSTGRES_USER=your_os_user
POSTGRES_PASSWORD=
```

### Database Setup

```bash
# Create the database
createdb spx_ai

# Run migrations
alembic upgrade head
```

See `POSTGRESQL_SETUP.md` for detailed PostgreSQL configuration steps.

### Start the Server

```bash
# Using the convenience script
./start_api.sh

# Or directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Data Pipeline

The system requires 1-minute SPX price data and SPXW options quote data stored as Parquet files. Data is sourced from ThetaData Terminal.

### File Naming Convention

```
data/processed/parquet_1m/
├── SPX_index_price_1m_YYYYMMDD.parquet       # SPX index prices
├── SPXW_option_quotes_1m_YYYYMMDD_expYYYYMMDD_sr200.parquet  # Options chain
└── ...
```

### Parquet Schemas

**SPX price file** (`SPX_index_price_1m_*.parquet`):

| Column    | Type    | Example                  |
|-----------|---------|--------------------------|
| timestamp | object  | `2025-01-02T09:30:00`    |
| price     | float32 | `4783.83`                |

**Options file** (`SPXW_option_quotes_1m_*.parquet`):

| Column        | Type     | Notes                     |
|---------------|----------|---------------------------|
| symbol        | category | `SPXW`                    |
| expiration    | category | `2025-01-02`              |
| strike        | float32  | `4750.0`                  |
| right         | category | `CALL` or `PUT`           |
| timestamp     | category | `2025-01-02T09:31:00`     |
| bid_size      | uint16   |                           |
| bid_exchange  | category | `"5"` (int cast to str)   |
| bid           | float32  |                           |
| bid_condition | int64    |                           |
| ask_size      | uint16   |                           |
| ask_exchange  | category |                           |
| ask           | float32  |                           |
| ask_condition | int64    |                           |

### Converting Existing CSVs to Parquet

If you have ThetaData CSV tick files already downloaded:

```bash
# Convert all CSV files found under /path/to/data, skip already converted
python scripts/csv_to_parquet.py \
    --source-dir /Users/you/Trading/data \
    --output-dir data/processed/parquet_1m \
    --skip-existing

# Dry run first to preview what will be converted
python scripts/csv_to_parquet.py --dry-run

# Force re-convert everything
python scripts/csv_to_parquet.py
```

The script recursively discovers all `SPX_index_price_tick_*.csv` and `SPXW_option_quotes_tick_*.csv` files, maps `_tick_` → `_1m_` in the filename, and writes the Parquet output with exact dtype matching.

### Downloading New Data

To download fresh data from a running ThetaData Terminal instance:

```bash
# Download a specific date
python scripts/download_thetadata.py --date 2025-03-21

# Download last N trading days
python scripts/download_thetadata.py --days-back 5

# Download a full date range
python scripts/download_thetadata.py \
    --start-date 2025-01-01 \
    --end-date 2025-12-31

# Skip dates already downloaded
python scripts/download_thetadata.py --days-back 30 --skip-existing

# Use a non-default terminal port
python scripts/download_thetadata.py --terminal-url http://localhost:25510 --days-back 5
```

**Current data coverage:** ~288 trading days, Dec 2024 → Feb 2026.

---

## API Reference

### REST Endpoints

#### System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/api/v1/status` | System status and available data range |

#### Backtests

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/backtest/start` | Start a new backtest |
| GET | `/api/v1/backtest` | List all backtests |
| GET | `/api/v1/backtest/{id}/status` | Get backtest status |
| GET | `/api/v1/backtest/{id}/results` | Get all trade results |
| DELETE | `/api/v1/backtest/{id}` | Cancel a running backtest |

#### Database (historical queries)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/database/backtests` | Paginated list of persisted backtests |
| GET | `/api/v1/database/backtests/{id}` | Full backtest record |
| GET | `/api/v1/database/backtests/{id}/trades` | All trades for a backtest |
| GET | `/api/v1/database/backtests/{id}/trades/{tid}/checkpoints` | Per-bar monitoring snapshots |
| DELETE | `/api/v1/database/backtests/{id}` | Delete backtest and all trades |

#### WebSocket

```
WS /ws/{client_id}
```

Real-time event stream during a backtest. Events:

```json
{ "type": "trade_result", "data": { ... } }
{ "type": "progress",     "data": { "completed": 45, "total": 60 } }
{ "type": "completed",    "data": { "backtest_id": "..." } }
{ "type": "error",        "data": { "message": "..." } }
```

### BacktestRequest Body

```json
{
  "mode": "date_range",
  "strategy": "ic_credit_spreads",
  "start_date": "2025-01-02",
  "end_date": "2025-06-30",
  "target_credit": 0.40,
  "spread_width": 10,
  "contracts": 1,
  "take_profit": 0.05,
  "stop_loss": 6.0,
  "monitor_interval": 1
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | enum | required | `single_day` \| `date_range` \| `live_simulation` |
| `strategy` | enum | `iron_condor` | `iron_condor` \| `credit_spreads` \| `ic_credit_spreads` |
| `start_date` | date | — | Used with `date_range` |
| `end_date` | date | — | Used with `date_range` |
| `single_date` | date | — | Used with `single_day` |
| `target_credit` | float | `0.40` | Minimum net credit per share per spread |
| `spread_width` | int | `10` | Strike width of each spread (points) |
| `contracts` | int | `1` | Number of contracts per position |
| `take_profit` | float | `0.05` | Exit when cost-to-close drops to this value ($/share) |
| `stop_loss` | float | `6.00` | Exit when cost-to-close reaches this value ($/share) |
| `monitor_interval` | int | `5` | Minutes between position checks (1 = every bar) |

---

## Strategy Modes

### `iron_condor`

Sells a put spread and a call spread simultaneously, creating a defined-risk position that profits when SPX stays within a range. Only enters when market conditions are neutral (low trend momentum). Never enters when market is trending significantly.

### `credit_spreads`

Direction-agnostic spread selling. The system picks either a put spread (neutral-to-bullish) or call spread (neutral-to-bearish) based on RSI, Bollinger Band position, and trend signals. Up to one put spread and one call spread can be open concurrently.

### `ic_credit_spreads`

Hybrid mode. Prefers an Iron Condor when market is neutral. If the market is trending, falls back to a single credit spread on the safe side (call spread on down-trending days, put spread on up-trending days). This gives more trade opportunities than `iron_condor` alone while maintaining directional bias awareness.

---

## Intraday Scan Engine

The engine (`enhanced_multi_strategy.py`) runs a minute-by-minute scan loop each trading day.

### Scan Loop Timeline

```
09:30  Market open
09:35  Scan loop begins (first 5-min bar complete)
  │    Every 1 minute:
  │      1. Update + check open position decay
  │      2. Check drift/trend guards
  │      3. If before 14:00 → evaluate entry signal
  │      4. If signal confirmed → try to open position
  │
14:00  Last new entry cutoff
  │    Existing positions continue to be monitored
  │
16:00  Hard close — all remaining positions force-closed
```

### Key Constants

```python
ENTRY_SCAN_START    = "09:35:00"  # First bar checked for entry
LAST_ENTRY_TIME     = "14:00:00"  # No new entries at or after 14:00
FINAL_EXIT_TIME     = "16:00:00"  # Hard close
MIN_DISTANCE_IC     = 50.0        # Min pts from SPX to IC short strikes
MIN_DISTANCE_SPREAD = 25.0        # Min pts from SPX to spread short strike
```

### Entry Signal

At each bar the engine:

1. Fetches the last 30 minutes of SPX prices
2. Runs technical indicators: RSI(14), 20-period Bollinger Bands, VWAP, linear trend
3. Selects a strategy type based on the combined signal
4. Applies all risk guards (see below)
5. If allowed, selects strikes targeting `target_credit` and opens the position

### Risk Guards

#### Trend Filter

Blocks Iron Condor entries when market is trending strongly:

```
TREND_FILTER_LOOKBACK_MINUTES = 30  bars
TREND_FILTER_POINTS           = 30  points directional move threshold
```

If SPX has moved more than 30 pts in one direction over the last 30 minutes, the trend filter fires and blocks any new IC. In `ic_credit_spreads` mode it redirects to a single spread on the safe side instead.

#### Consecutive-Signal Tightening

```
TREND_TIGHTEN_CONSECUTIVE = 3   # After 3 consecutive signals
TREND_TIGHTEN_FACTOR      = 0.5 # Threshold cut in half
```

After 3 consecutive bars pointing the same direction, the trend threshold is halved, making the filter more sensitive.

#### Daily Drift Guard (Sticky)

The most important protective mechanism. Anchors to the SPX price at market open (first valid bar among 09:30, 09:31, 09:32, 09:35) and tracks cumulative drift all day:

```
DRIFT_BLOCK_POINTS   = 15.0   # If |drift| >= 15 pts → block put spreads + IC
DRIFT_DELAY_POINTS   = 12.0   # If |drift| >= 12 pts AND time < 12:00 → delay IC
DRIFT_MIN_ENTRY_HOUR = "12:00:00"
```

Flags are **sticky** (latched True, never reset):
- `_put_spread_ever_blocked` — latches True when `|drift| >= 15`. Blocks put spread entries all day, even if the market bounces back.
- `_ic_ever_blocked` — latches True when `|drift| >= 15` OR when `|drift| >= 12` before noon.

Call spreads are **never blocked by drift** — on a down-trending day, selling call spreads is the safe side.

**Why sticky flags matter:** Without a latch, a temporary bounce (e.g. SPX drops 19 pts at 09:59 then recovers to 13 pts below open by 10:26) would re-enable the guard mid-session, allowing a put spread to open right before the market drops further.

#### Dynamic Minimum Distance (IC only)

```python
morning_range = spx_history.max() - spx_history.min()
dynamic_min_dist = max(MIN_DISTANCE_IC, morning_range * 0.75)
```

When volatility is high in the morning session, IC short strikes are pushed further from SPX proportionally. Single credit spreads always use the flat `MIN_DISTANCE_SPREAD = 25` floor.

### Position Decay Monitoring (`IntradayPositionMonitor`)

Positions are monitored every `monitor_interval` minutes. Exit conditions:

| Strategy | Exit trigger |
|----------|-------------|
| Credit spread | Cost-to-close ≤ `take_profit` (default 0.05/share) |
| Credit spread | Cost-to-close ≥ `stop_loss` (default 6.00/share) |
| Iron Condor (put side) | Put spread cost ≤ `take_profit` |
| Iron Condor (call side) | Call spread cost ≤ `take_profit` |
| Iron Condor (either side) | Either side cost ≥ `stop_loss` |

IC legs are managed **independently**. Each side closes separately when it decays to the take-profit level. The other side continues to be monitored.

### Concurrent Position Limits

| Scenario | Allowed |
|----------|---------|
| IC open + new IC signal | No (one IC per day) |
| IC open + spread signal | No (IC occupies both sides) |
| Put spread open + call spread signal | Yes |
| Call spread open + put spread signal | Yes (if drift permits) |
| Put spread open + new put spread | No |

---

## Database Schema

All backtests and trades are persisted to PostgreSQL immediately after each trade completes (not buffered to end-of-session).

### Tables

**`backtest_runs`** — One row per backtest execution

| Column | Type | Notes |
|--------|------|-------|
| id | UUID | Primary key |
| backtest_id | String | Unique, indexed |
| mode | String | single_day / date_range / live_simulation |
| strategy_type | String | iron_condor / credit_spreads / ic_credit_spreads |
| start_date / end_date | Date | Date range |
| target_delta | Float | Strike delta target |
| spread_width | Integer | Points |
| decay_threshold | Float | Take-profit level |
| monitor_interval | Integer | Minutes |
| status | String | pending / running / completed / failed / cancelled |
| total_trades | Integer | |
| successful_trades | Integer | |
| total_pnl | Float | |
| win_rate | Float | |
| created_at / completed_at | DateTime | |
| parameters | JSONB | Full request parameters |

**`trades`** — One row per trade executed

| Column | Type | Notes |
|--------|------|-------|
| id | UUID | Primary key |
| trade_id | String | Unique |
| backtest_run_id | UUID | FK → backtest_runs |
| trade_date | Date | Indexed |
| entry_time / exit_time | String | HH:MM:SS |
| entry_spx_price / exit_spx_price | Float | |
| strategy_type | String | |
| strikes | JSONB | Full strike details |
| entry_credit / exit_cost | Float | Per-share values |
| pnl / pnl_percentage | Float | |
| exit_reason | String | decay_threshold / stop_loss / force_close / etc. |
| is_winner | Boolean | |
| monitoring_data | JSONB | Per-bar checkpoints for charting |

### Migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Create a new migration after changing ORM models
alembic revision --autogenerate -m "describe change"
```

---

## Running Backtests

### Via the API (recommended)

```bash
# Start a single-day backtest
curl -X POST http://localhost:8000/api/v1/backtest/start \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "single_day",
    "strategy": "ic_credit_spreads",
    "single_date": "2026-02-09",
    "take_profit": 0.05,
    "monitor_interval": 1
  }'

# Check status
curl http://localhost:8000/api/v1/backtest/{backtest_id}/status

# Get results
curl http://localhost:8000/api/v1/backtest/{backtest_id}/results
```

### Direct CLI

The engine can also be run directly for debugging:

```bash
# Single day with default settings
python enhanced_multi_strategy.py --date 2026-02-09

# Full date range with a specific strategy
python enhanced_multi_strategy.py \
    --start-date 2026-02-09 \
    --end-date 2026-02-13 \
    --strategy ic_credit_spreads

# Show per-bar monitoring output
python enhanced_multi_strategy.py --date 2026-02-09 --show-monitoring
```

---

## Testing

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run with coverage
python -m pytest tests/unit/ --cov=. --cov-report=term-missing

# Run a specific test file
python -m pytest tests/unit/test_delta_strike_selector.py -v
```

The unit test suite covers:

- `test_delta_strike_selector.py` — Strike selection, `IntradayPositionMonitor` decay logic, IC leg independence
- `test_technical_analysis.py` — RSI, Bollinger Bands, VWAP, trend detection

---

## Dependencies

```
fastapi>=0.104.0        Web framework
uvicorn[standard]       ASGI server
websockets>=12.0        WebSocket support
sqlalchemy>=2.0.0       ORM
alembic>=1.12.0         Database migrations
psycopg2-binary>=2.9.0  PostgreSQL driver
pandas>=2.0.0           Data processing
pyarrow>=10.0.0         Parquet I/O
numpy>=1.24.0           Numerical operations
scipy>=1.10.0           Statistical functions (Black-Scholes)
scikit-learn>=1.3.0     Used for regression/indicator helpers
pydantic>=2.0.0         Request/response validation
loguru>=0.7.0           Structured logging
python-dotenv>=1.0.0    Environment variable loading
requests>=2.31.0        ThetaData REST client
```

---

## Important Notes

1. **Backtesting only** — This system is for research and backtesting. It does not place live orders.
2. **0DTE focus** — Designed for same-day expiration SPXW options (09:30–16:00 ET).
3. **Data required locally** — All data is read from local Parquet files. The ThetaData Terminal must be running to download new data; it is not needed for backtesting.
4. **Options pricing** — The system reads bid/ask from historical option quote data. Greek calculations (delta for strike selection) use Black-Scholes via `scipy`.
5. **Paper trading results** — Historical P&L calculations assume fills at mid-price with no slippage. Real-world results will differ.
