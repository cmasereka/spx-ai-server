"""
Live Trading API Routes (IBKR paper / live)

Endpoints:
  POST   /api/v1/trading/sessions                — start a new IBKR session
  GET    /api/v1/trading/sessions                — list all sessions
  GET    /api/v1/trading/sessions/{id}           — get session status
  DELETE /api/v1/trading/sessions/{id}           — stop a running session
  DELETE /api/v1/trading/sessions/{id}?purge=true — stop and permanently delete
  POST   /api/v1/trading/test-connection         — test IBKR connectivity
  POST   /api/v1/trading/diagnose-market-data    — test SPX price + options chain fetch
"""

import asyncio
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from .models import LiveTradingRequest, LiveTradingStatus, IBKRConnectionConfig, IBKRDiagnosticRequest, DiagnosticRequest, BrokerEnum
from .live_trading_service import LiveTradingService
from .websocket_manager import WebSocketManager

router = APIRouter(prefix="/api/v1/trading", tags=["Live Trading"])

# Injected by main.py after services are initialised
live_trading_service: LiveTradingService = None   # type: ignore[assignment]
websocket_manager: WebSocketManager = None        # type: ignore[assignment]


def init_router(service: LiveTradingService, ws_manager: WebSocketManager):
    """Call this from main.py lifespan once services are ready."""
    global live_trading_service, websocket_manager
    live_trading_service = service
    websocket_manager = ws_manager


@router.post("/sessions", response_model=dict, status_code=201)
async def start_live_trading_session(request: LiveTradingRequest):
    """
    Start a new IBKR paper / live trading session.

    Set **ibkr.port** to:
    - `7497` for TWS paper trading (default)
    - `7496` for TWS live trading
    - `4002` for IB Gateway paper trading
    - `4001` for IB Gateway live trading

    The session will attempt to connect to IBKR immediately.  If the
    connection fails, the session status will be set to `failed`.
    """
    try:
        session_id = await live_trading_service.start_session(request, websocket_manager)
        return {
            "session_id": session_id,
            "status": "connecting",
            "message": "Live trading session initiated — connecting to IBKR",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to start live trading session: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sessions", response_model=list)
async def list_live_trading_sessions():
    """List all live trading sessions (most recent first)."""
    try:
        sessions = live_trading_service.list_sessions()
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return [s.dict() for s in sessions]
    except Exception as exc:
        logger.error(f"Failed to list live trading sessions: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sessions/{session_id}", response_model=LiveTradingStatus)
async def get_live_trading_session(session_id: str):
    """Get the current status of a live trading session."""
    status = live_trading_service.get_session(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    return status


@router.delete("/sessions/{session_id}")
async def stop_live_trading_session(
    session_id: str,
    purge: bool = Query(False, description="If true, permanently delete the session record"),
):
    """
    Stop a running session and close all open IBKR positions.
    Pass ``?purge=true`` to also permanently remove the session record.
    """
    try:
        if purge:
            deleted = await live_trading_service.delete_session(session_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Session not found")
            return {"message": "Session deleted", "session_id": session_id}
        else:
            stopped = await live_trading_service.stop_session(session_id)
            if not stopped:
                raise HTTPException(status_code=404, detail="Session not found")
            return {"message": "Session stopped", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to stop/delete live session {session_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/test-connection", response_model=dict)
async def test_ibkr_connection(config: IBKRConnectionConfig):
    """
    Test IBKR TWS / IB Gateway connectivity without starting a trading session.

    Attempts to connect, retrieves the account summary, then disconnects.
    Returns connection status, account info, and the next valid order ID
    so you can confirm the API is fully functional.

    Use this before starting a live session to diagnose connection issues.
    """
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _test_ibkr_connection_sync, config
        )
        return result
    except Exception as exc:
        logger.error(f"IBKR connection test failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


def _test_ibkr_connection_sync(config: IBKRConnectionConfig) -> dict:
    """Blocking IBKR connection test — runs in a thread pool."""
    import asyncio
    import traceback

    # ib_insync's synchronous helpers call asyncio.get_event_loop() internally.
    # ThreadPoolExecutor worker threads have no event loop by default, so we
    # must create and set one before touching any ib_insync code.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        from ib_insync import IB
    except ImportError:
        return {
            "connected": False,
            "error": "ib_insync is not installed. Run: pip install ib_insync",
        }

    ib = IB()
    try:
        ib.connect(config.host, config.port, clientId=config.client_id, timeout=10)

        if not ib.isConnected():
            return {
                "connected": False,
                "host": config.host,
                "port": config.port,
                "error": "Connection returned but isConnected() is False",
            }

        # Fetch account summary to confirm API is fully operational
        account = config.account or (ib.managedAccounts()[0] if ib.managedAccounts() else "")
        summary = ib.accountSummary(account)
        next_order_id = ib.client.getReqId()

        fields = {}
        for item in summary:
            if item.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower", "AccountType"):
                fields[item.tag] = item.value

        return {
            "connected": True,
            "host": config.host,
            "port": config.port,
            "client_id": config.client_id,
            "account": account,
            "next_order_id": next_order_id,
            "account_summary": fields,
        }

    except Exception as exc:
        error_type = type(exc).__name__
        error_detail = str(exc)
        tb = traceback.format_exc()
        logger.error(f"IBKR connection test [{error_type}]: {error_detail}")
        return {
            "connected": False,
            "host": config.host,
            "port": config.port,
            "error_type": error_type,
            "error": error_detail,
            "traceback": tb,
        }
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass
        loop.close()


# ---------------------------------------------------------------------------
# Market data diagnostic endpoint
# ---------------------------------------------------------------------------

@router.post("/diagnose-market-data", response_model=dict)
async def diagnose_market_data(request: DiagnosticRequest):
    """
    Test the full SPX-price + options-chain data path without starting a trading session.

    Set broker='ibkr' to test IBKR connectivity.
    Set broker='tastytrade' to test TastyTrade authentication, option chain, and DXLink streaming.
    """
    try:
        if request.broker == BrokerEnum.TASTYTRADE:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _diagnose_tastytrade_sync, request
            )
        else:
            # Convert to legacy IBKRDiagnosticRequest shape for the existing function
            ibkr_req = IBKRDiagnosticRequest(
                ibkr=request.ibkr,
                trade_date=request.trade_date,
                spx_price_hint=request.spx_price_hint,
                diagnostic_client_id=request.diagnostic_client_id,
                num_strikes=request.num_strikes,
            )
            result = await asyncio.get_event_loop().run_in_executor(
                None, _diagnose_market_data_sync, ibkr_req
            )
        return result
    except Exception as exc:
        logger.error(f"Market data diagnostic failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


def _diagnose_market_data_sync(request: IBKRDiagnosticRequest) -> dict:
    """
    Blocking diagnostic — runs in a thread pool.

    Steps:
      1. Connect to IBKR with the diagnostic clientId
      2. Detect market-hours status
      3. Fetch SPX Index quote (last / close price)
      4. Batch-request SPXW options chain, wait 3 s for snapshots to arrive
      5. Return a structured diagnostic report
    """
    import asyncio as _asyncio
    import traceback
    from datetime import datetime, date as _date, time as _time

    loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(loop)

    try:
        from ib_insync import IB, Index, Option
    except ImportError:
        return {
            "ok": False,
            "error": "ib_insync is not installed. Run: pip install ib_insync",
        }

    cfg = request.ibkr
    trade_date = request.trade_date or _date.today().strftime("%Y-%m-%d")
    expiry_str = datetime.strptime(trade_date, "%Y-%m-%d").strftime("%Y%m%d")

    # Determine market-hours status in US/Eastern time
    import pytz
    now_et = datetime.now(pytz.timezone("America/New_York"))
    market_open  = _time(9, 30)
    market_close = _time(16, 0)
    is_market_hours = market_open <= now_et.time() <= market_close

    report: dict = {
        "ok": False,
        "connected": False,
        "host": cfg.host,
        "port": cfg.port,
        "diagnostic_client_id": request.diagnostic_client_id,
        "trade_date": trade_date,
        "server_time": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "is_market_hours": is_market_hours,
        "spx": {},
        "options": {},
        "errors": [],
    }

    ib = IB()
    try:
        ib.connect(cfg.host, cfg.port,
                   clientId=request.diagnostic_client_id, timeout=10)
        if not ib.isConnected():
            report["errors"].append("isConnected() returned False after connect()")
            return report

        report["connected"] = True
        ib.reqMarketDataType(1)  # Live data

        # ------------------------------------------------------------------
        # 1. SPX Index price
        #
        # snapshot=True is unreliable for Index contracts — IBKR sometimes
        # returns nothing.  Use a live subscription (snapshot=False), wait
        # long enough for IBKR to push the current/last-known price, then
        # cancel.  This works both during and outside market hours.
        # ------------------------------------------------------------------
        spx_contract = Index("SPX", "CBOE", "USD")
        ib.qualifyContracts(spx_contract)

        spx_ticker = ib.reqMktData(spx_contract, "", snapshot=False, regulatorySnapshot=False)
        ib.sleep(3)   # live subscription: wait for first price tick

        spx_price = None
        spx_source = "none"
        if spx_ticker.last and spx_ticker.last > 0:
            spx_price = float(spx_ticker.last)
            spx_source = "last (live)"
        elif spx_ticker.close and spx_ticker.close > 0:
            spx_price = float(spx_ticker.close)
            spx_source = "close (previous session)"
        elif spx_ticker.bid and spx_ticker.bid > 0:
            spx_price = float(spx_ticker.bid)
            spx_source = "bid"

        report["spx"] = {
            "price":  spx_price,
            "source": spx_source,
            "bid":    float(spx_ticker.bid)   if spx_ticker.bid   and spx_ticker.bid   > 0 else None,
            "ask":    float(spx_ticker.ask)   if spx_ticker.ask   and spx_ticker.ask   > 0 else None,
            "last":   float(spx_ticker.last)  if spx_ticker.last  and spx_ticker.last  > 0 else None,
            "close":  float(spx_ticker.close) if spx_ticker.close and spx_ticker.close > 0 else None,
        }

        ib.cancelMktData(spx_contract)

        # Use spx_price_hint as fallback, but ignore 0 / near-zero values sent
        # by frontends that submit an empty field as 0.  Default to 5800.
        hint = request.spx_price_hint if request.spx_price_hint > 100 else 5800.0
        center = spx_price or hint

        if not spx_price:
            report["errors"].append(
                f"No SPX price returned by IBKR. "
                f"Falling back to center={center} (from {'spx_price_hint' if request.spx_price_hint > 100 else 'built-in default 5800'}) "
                f"for the options chain query."
            )

        # ------------------------------------------------------------------
        # 2. SPXW options chain — batch all requests then wait once
        # ------------------------------------------------------------------
        n = request.num_strikes
        base = round(center / 5) * 5           # snap to nearest 5-pt interval
        strikes = [base + (i - n) * 5 for i in range(n * 2 + 1)]

        # Submit all snapshot requests up-front
        pending: list = []   # (strike, right, contract, ticker)
        for strike in strikes:
            for right in ("P", "C"):
                contract = Option("SPX", expiry_str, strike, right,
                                  exchange="CBOE", currency="USD",
                                  multiplier="100",
                                  tradingClass="SPXW")
                try:
                    ticker = ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
                    pending.append((strike, right, contract, ticker))
                except Exception as e:
                    report["errors"].append(f"reqMktData {strike}{right}: {e}")

        # Wait for IBKR to push all snapshot responses (3 s is sufficient in practice)
        ib.sleep(3)

        options_checked = 0
        options_with_data = 0
        sample_rows = []
        option_errors = []

        for strike, right, contract, ticker in pending:
            try:
                bid   = float(ticker.bid)   if ticker.bid   and ticker.bid   > 0 else None
                ask   = float(ticker.ask)   if ticker.ask   and ticker.ask   > 0 else None
                last  = float(ticker.last)  if ticker.last  and ticker.last  > 0 else None
                close = float(ticker.close) if ticker.close and ticker.close > 0 else None

                options_checked += 1
                has_data = any(v is not None for v in (bid, ask, last, close))
                if has_data:
                    options_with_data += 1

                if len(sample_rows) < 8 and has_data:
                    mid = round((bid + ask) / 2, 2) if bid and ask else None
                    greeks = ticker.modelGreeks
                    sample_rows.append({
                        "strike": strike,
                        "right": right,
                        "bid":   bid,
                        "ask":   ask,
                        "last":  last,
                        "close": close,
                        "mid":   mid,
                        "delta": round(float(greeks.delta),      4) if greeks and greeks.delta      else None,
                        "gamma": round(float(greeks.gamma),      6) if greeks and greeks.gamma      else None,
                        "theta": round(float(greeks.theta),      4) if greeks and greeks.theta      else None,
                        "iv":    round(float(greeks.impliedVol), 4) if greeks and greeks.impliedVol else None,
                    })

                # snapshot=True subscriptions are auto-cancelled by IBKR — do NOT call cancelMktData()

            except Exception as opt_err:
                option_errors.append(f"{strike}{right}: {opt_err}")

        if option_errors:
            report["errors"].extend(option_errors[:5])

        report["options"] = {
            "expiry":              expiry_str,
            "center_strike":       round(center),
            "strikes_queried":     len(pending),
            "strikes_checked":     options_checked,
            "strikes_with_data":   options_with_data,
            "data_coverage_pct":   round(options_with_data / options_checked * 100, 1) if options_checked else 0,
            "sample":              sample_rows,
        }

        # ------------------------------------------------------------------
        # 3. Assessment
        # ------------------------------------------------------------------
        report["ok"] = report["connected"] and (spx_price is not None)

        if options_with_data > 0 and spx_price:
            report["assessment"] = (
                f"Full data path healthy. "
                f"SPX={spx_price} ({spx_source}). "
                f"{options_with_data}/{options_checked} options had quotes."
            )
        elif spx_price and not is_market_hours:
            report["assessment"] = (
                f"SPX={spx_price} ({spx_source}) — connection is working. "
                f"0DTE SPXW options have no bid/ask quotes before market opens (09:30 ET). "
                f"Run again after 09:30 to confirm full options data flow."
            )
        elif spx_price and is_market_hours:
            report["assessment"] = (
                f"Market is open but options returned no data. "
                f"Check IBKR data subscriptions for SPXW/CBOE options. "
                f"SPX={spx_price} ({spx_source})."
            )
        elif not spx_price and not is_market_hours:
            report["assessment"] = (
                f"Connected to IBKR but SPX price not returned yet. "
                f"This can happen very early pre-market before IBKR pushes closing data. "
                f"Options also have no quotes outside market hours. "
                f"Try again closer to 09:30 ET."
            )
        else:
            report["assessment"] = (
                "Connected to IBKR but no data returned (SPX or options). "
                "Check IBKR data subscriptions and market data permissions."
            )

        return report

    except Exception as exc:
        report["errors"].append(f"Fatal: {exc}")
        report["errors"].append(traceback.format_exc())
        return report

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass
        loop.close()


def _diagnose_tastytrade_sync(request: DiagnosticRequest) -> dict:
    """
    Blocking TastyTrade diagnostic — runs in a thread pool.

    Steps:
      1. Create a Session (tests authentication)
      2. Fetch account (tests account access)
      3. Load SPXW option chain for trade_date (tests REST data access)
      4. Open DXLinkStreamer briefly and collect Quote + Greeks snapshots
         for a sample of ATM strikes (tests streaming connectivity)
      5. Return a structured diagnostic report
    """
    import asyncio as _asyncio
    import traceback
    from datetime import datetime, date as _date, time as _time
    import pytz

    loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(loop)

    now_et = datetime.now(pytz.timezone("America/New_York"))
    market_open  = _time(9, 30)
    market_close = _time(16, 0)
    is_market_hours = market_open <= now_et.time() <= market_close

    trade_date = request.trade_date or _date.today().strftime("%Y-%m-%d")
    try:
        target_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
    except ValueError:
        target_date = _date.today()

    cfg = request.tastytrade
    report: dict = {
        "ok": False,
        "broker": "tastytrade",
        "authenticated": False,
        "account_accessible": False,
        "trade_date": trade_date,
        "server_time": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "is_market_hours": is_market_hours,
        "spx": {},
        "options": {},
        "streaming": {},
        "errors": [],
    }

    try:
        from tastytrade import Session
        from tastytrade.account import Account
        from tastytrade.instruments import NestedOptionChain
        from tastytrade.streamer import DXLinkStreamer
        from tastytrade.dxfeed import Quote, Greeks
    except ImportError as e:
        report["errors"].append(f"tastytrade not installed: {e}")
        return report

    # ------------------------------------------------------------------
    # 1. Authentication
    # ------------------------------------------------------------------
    try:
        session = Session(
            cfg.provider_secret,
            cfg.refresh_token,
            is_test=cfg.is_paper,
        )
        report["authenticated"] = True
        report["environment"] = "paper (cert)" if cfg.is_paper else "live (prod)"
    except Exception as exc:
        report["errors"].append(f"Authentication failed: {exc}")
        return report

    # ------------------------------------------------------------------
    # 2. Account access
    # ------------------------------------------------------------------
    try:
        account = loop.run_until_complete(Account.get(session, cfg.account_number))
        report["account_accessible"] = True
        report["account_number"] = account.account_number
    except Exception as exc:
        report["errors"].append(f"Account access failed: {exc}")
        # Continue — still useful to test data

    # ------------------------------------------------------------------
    # 3. SPXW option chain (REST)
    # ------------------------------------------------------------------
    today_expir = None
    streamer_symbols: list = []
    sample_meta: list = []   # [{strike, option_type, call_sym, put_sym}]

    try:
        chains = loop.run_until_complete(NestedOptionChain.get(session, "SPXW"))

        # Find requested expiry
        for chain in chains:
            for expir in chain.expirations:
                if expir.expiration_date == target_date:
                    today_expir = expir
                    break
            if today_expir:
                break

        if today_expir is None:
            report["errors"].append(
                f"No SPXW option chain found for {trade_date}. "
                "Market may be closed or it is not a trading day."
            )
            report["options"]["expiry"] = trade_date
            report["options"]["strikes_loaded"] = 0
        else:
            all_strikes = today_expir.strikes
            total = len(all_strikes)

            # Collect streamer symbols for a centre sample (up to num_strikes each side)
            n = request.num_strikes
            mid_idx = total // 2
            lo = max(0, mid_idx - n)
            hi = min(total, mid_idx + n + 1)
            sample_strikes = all_strikes[lo:hi]

            for s in sample_strikes:
                if s.call_streamer_symbol:
                    streamer_symbols.append(s.call_streamer_symbol)
                    sample_meta.append({
                        "strike": float(s.strike_price),
                        "option_type": "call",
                        "streamer_symbol": s.call_streamer_symbol,
                    })
                if s.put_streamer_symbol:
                    streamer_symbols.append(s.put_streamer_symbol)
                    sample_meta.append({
                        "strike": float(s.strike_price),
                        "option_type": "put",
                        "streamer_symbol": s.put_streamer_symbol,
                    })

            # All symbols for counting
            all_syms = sum(
                (1 if s.call_streamer_symbol else 0) + (1 if s.put_streamer_symbol else 0)
                for s in all_strikes
            )
            report["options"] = {
                "expiry": trade_date,
                "total_strikes": total,
                "total_symbols": all_syms,
                "sample_strikes_requested": n,
                "sample_symbols_fetched": len(streamer_symbols),
            }
    except Exception as exc:
        report["errors"].append(f"Option chain fetch failed: {exc}")
        report["errors"].append(traceback.format_exc())

    # ------------------------------------------------------------------
    # 4. DXLink streaming — brief Quote + Greeks snapshot
    # ------------------------------------------------------------------
    async def _stream_snapshot():
        quotes: dict = {}    # sym → {bid, ask}
        greeks: dict = {}    # sym → {delta, iv}

        try:
            async with DXLinkStreamer(session) as streamer:
                if streamer_symbols:
                    await streamer.subscribe(Quote, streamer_symbols)
                    await streamer.subscribe(Greeks, streamer_symbols)

                # Collect events for up to 5 seconds (or 3 s outside market hours)
                wait_secs = 3.0 if is_market_hours else 2.0
                deadline = _asyncio.get_event_loop().time() + wait_secs

                while _asyncio.get_event_loop().time() < deadline:
                    remaining = deadline - _asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        event = await _asyncio.wait_for(
                            streamer.get_event(Quote), timeout=remaining
                        )
                        sym = event.event_symbol
                        if sym in set(streamer_symbols):
                            quotes[sym] = {
                                "bid": float(event.bid_price) if event.bid_price else None,
                                "ask": float(event.ask_price) if event.ask_price else None,
                            }
                    except _asyncio.TimeoutError:
                        break
                    except Exception:
                        break

                # Drain any buffered Greeks (non-blocking)
                for _ in range(len(streamer_symbols) * 2):
                    g = streamer.get_event_nowait(Greeks)
                    if g is None:
                        break
                    sym = g.event_symbol
                    greeks[sym] = {
                        "delta": float(g.delta) if g.delta is not None else None,
                        "iv": float(g.volatility) if g.volatility is not None else None,
                    }

        except Exception as exc:
            return {"error": str(exc)}, quotes, greeks

        return None, quotes, greeks

    streaming_error, quotes, greeks_data = loop.run_until_complete(_stream_snapshot())

    if streaming_error:
        report["errors"].append(f"DXLink streaming failed: {streaming_error['error']}")
        report["streaming"] = {"ok": False, "error": streaming_error["error"]}
    else:
        report["streaming"] = {
            "ok": True,
            "symbols_subscribed": len(streamer_symbols),
            "quotes_received": len(quotes),
            "greeks_received": len(greeks_data),
        }

        # Build sample rows with bid/ask/delta
        sample_rows = []
        for meta in sample_meta:
            sym = meta["streamer_symbol"]
            q = quotes.get(sym, {})
            g = greeks_data.get(sym, {})
            bid = q.get("bid")
            ask = q.get("ask")
            mid = round((bid + ask) / 2, 2) if bid and ask else None
            sample_rows.append({
                "strike":      meta["strike"],
                "option_type": meta["option_type"],
                "bid":         bid,
                "ask":         ask,
                "mid":         mid,
                "delta":       round(g["delta"], 4) if g.get("delta") is not None else None,
                "iv":          round(g["iv"], 4)    if g.get("iv")    is not None else None,
            })
        report["options"]["sample"] = sample_rows

        # Infer approximate SPX price from near-ATM put/call mid prices
        atm_rows = [r for r in sample_rows if r["mid"] is not None]
        if atm_rows:
            hint = request.spx_price_hint if request.spx_price_hint > 100 else 5800.0
            report["spx"] = {
                "price": None,
                "note": "SPX spot not directly available outside streaming session. "
                        f"Using hint={hint} for chain centre.",
                "spx_price_hint": hint,
            }
        else:
            report["spx"] = {
                "price": None,
                "note": "No option quotes received — market may be closed.",
            }

    # ------------------------------------------------------------------
    # 5. Assessment
    # ------------------------------------------------------------------
    options_ok = report["options"].get("total_symbols", 0) > 0
    streaming_ok = report["streaming"].get("ok", False)
    quotes_ok = report["streaming"].get("quotes_received", 0) > 0

    if report["authenticated"] and options_ok and streaming_ok and (quotes_ok or not is_market_hours):
        report["ok"] = True
        if quotes_ok:
            report["assessment"] = (
                f"Full data path healthy. "
                f"Auth ✓  Account ✓  Option chain: {report['options']['total_symbols']} symbols ✓  "
                f"DXLink: {report['streaming']['quotes_received']} quotes received ✓"
            )
        else:
            report["assessment"] = (
                f"Connection healthy (market closed). "
                f"Auth ✓  Account ✓  Option chain: {report['options']['total_symbols']} symbols ✓  "
                f"DXLink connected but no quotes yet (expected outside market hours)."
            )
    elif report["authenticated"] and options_ok and not streaming_ok:
        report["ok"] = False
        report["assessment"] = (
            "Auth and option chain OK but DXLink streaming failed. "
            "Check network/firewall for WebSocket connections to TastyTrade."
        )
    elif report["authenticated"] and not options_ok:
        report["ok"] = False
        report["assessment"] = (
            f"Auth OK but no SPXW option chain found for {trade_date}. "
            "Market may be closed or credentials lack data permissions."
        )
    else:
        report["ok"] = False
        report["assessment"] = "Authentication failed. Check provider_secret and refresh_token."

    loop.close()
    return report
