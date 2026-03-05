"""
LiveTradingLoop — broker-agnostic standalone real-time trading loop.

Works with any MarketDataProvider / BrokerAdapter pair (IBKR, TastyTrade, etc.).
Each of the 6 core steps (SPX price → drift update → evaluate entry →
send order → monitor position → close) is a named method with step-number
logging.  The main loop in run_day() reads top-to-bottom with no hidden
engine-state swaps beyond the 3 needed for strategy building.

Replaces LiveTradingSession for all broker types.
"""

import sys
sys.path.append('.')

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from broker.adapter import BrokerAdapter, OrderResult
from broker.null_adapter import NullBrokerAdapter
from market_data.provider import MarketDataProvider

# Re-use the shim from session.py so strategy_builder price updates work
from trading.session import _LiveQueryEngineShim

from enhanced_backtest import (
    StrategyType, MarketSignal, TechnicalAnalyzer, TechnicalIndicators,
    IronCondorLegStatus, EnhancedBacktestResult, StrikeSelection,
    DayBacktestResult,
)
from delta_strike_selector import (
    DeltaStrikeSelector, IntradayPositionMonitor, IronCondorStrikeSelection,
)
from enhanced_multi_strategy import (
    EnhancedBacktestingEngine,
    _build_minute_grid,
    ENTRY_SCAN_START, LAST_ENTRY_TIME, FINAL_EXIT_TIME,
    MIN_DISTANCE_IC, MIN_DISTANCE_SPREAD,
    DRIFT_BLOCK_POINTS, DRIFT_IC_BLOCK_POINTS,
    PUT_SPREAD_MAX_RSI_ON_NEG_DRIFT, PUT_SPREAD_MAX_RSI_ON_POS_DRIFT,
    INTRADAY_CALL_REVERSAL_POINTS, INTRADAY_PUT_REVERSAL_POINTS,
    CALL_SPREAD_MAX_ENTRY_RSI,
    STRATEGY_IRON_CONDOR, STRATEGY_CREDIT_SPREADS, STRATEGY_IC_CREDIT_SPREADS,
    PUT_SPREAD_DRIFT_CONFIRM_MINUTES,
)


# ---------------------------------------------------------------------------
# Drift guard state
# ---------------------------------------------------------------------------

@dataclass
class _DriftGuards:
    """Drift-based entry guard state for one trading day."""
    spx_open: Optional[float]
    put_spread_ever_blocked: bool = False
    call_spread_ever_blocked: bool = False
    ic_ever_blocked: bool = False
    call_blocked_latch_time: Optional[str] = None
    intraday_max_drift: float = 0.0
    intraday_min_drift: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LiveTradingLoop:
    """
    Standalone real-time trading loop for IBKR connections.

    Replaces the engine-state-swap approach used by LiveTradingSession for
    IBKR.  Each of the 6 core steps is a named method; the main loop in
    run_day() reads top-to-bottom with clear per-step logging.

    TastyTrade sessions: use LiveTradingSession (trading/session.py).
    """

    def __init__(
        self,
        engine: EnhancedBacktestingEngine,
        market_data_provider: MarketDataProvider,
        broker_adapter: Optional[BrokerAdapter] = None,
    ):
        self._engine   = engine
        self._provider = market_data_provider
        self._broker   = broker_adapter or NullBrokerAdapter()

        # Fresh DeltaStrikeSelector using the live provider directly —
        # no engine.delta_selector swap needed.
        self._delta_selector = DeltaStrikeSelector(
            market_data_provider, engine.ic_loader
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_day(
        self,
        date: str,
        config,                  # LiveTradingRequest
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> DayBacktestResult:
        """
        Run one full trading day and return a DayBacktestResult.

        Blocks until market close for real-time providers; completes
        instantly for simulation providers.

        Performs 3 engine sub-component swaps for the duration of the day:
          1. engine.enhanced_query_engine → live provider (for _build_* methods)
          2. engine.strategy_builder.query_engine → _LiveQueryEngineShim
          3. engine.strategy_builder.data_adapter.query_engine → same shim
        All three are restored in the finally block.
        """
        logger.info(
            f"LiveTradingLoop.run_day: {date} | mode={config.strategy.value} "
            f"credit={config.target_credit} qty={config.contracts} "
            f"tp={config.take_profit} sl={config.stop_loss}"
        )

        def _fire(event: Dict[str, Any]):
            if progress_callback:
                try:
                    progress_callback(event)
                except Exception as _e:
                    logger.debug(f"progress_callback error (ignored): {_e}")

        live_shim         = _LiveQueryEngineShim(self._provider)
        orig_qe           = self._engine.enhanced_query_engine
        orig_sb_qe        = self._engine.strategy_builder.query_engine
        orig_sb_da_qe     = self._engine.strategy_builder.data_adapter.query_engine

        try:
            self._engine.enhanced_query_engine                      = self._provider
            self._engine.strategy_builder.query_engine              = live_shim
            self._engine.strategy_builder.data_adapter.query_engine = live_shim
            return self._run_day_inner(date, config, _fire)
        finally:
            self._engine.enhanced_query_engine                      = orig_qe
            self._engine.strategy_builder.query_engine              = orig_sb_qe
            self._engine.strategy_builder.data_adapter.query_engine = orig_sb_da_qe

    # ------------------------------------------------------------------
    # Inner loop (called after engine sub-components are swapped)
    # ------------------------------------------------------------------

    def _run_day_inner(self, date: str, config, fire) -> DayBacktestResult:
        strategy_mode = config.strategy.value
        quantity      = config.contracts
        entry_start   = config.entry_start_time
        last_entry    = config.last_entry_time
        skip_ind      = config.skip_indicators

        # Build per-run monitor; strategy_builder already points to live shim.
        monitor = IntradayPositionMonitor(
            self._provider,
            self._engine.strategy_builder,
            take_profit       = config.take_profit,
            stop_loss         = config.stop_loss,
            monitor_interval  = config.monitor_interval,
            stale_loss_minutes    = config.stale_loss_minutes,
            stale_loss_threshold  = config.stale_loss_threshold,
            stagnation_window     = config.stagnation_window,
            min_improvement       = config.min_improvement,
        )

        # Fetch SPX opening price for drift guards
        spx_open: Optional[float] = None
        for _t0 in ("09:30:00", "09:31:00", "09:32:00", "09:35:00"):
            _p = self._provider.get_fastest_spx_price(date, _t0)
            if _p and _p > 0:
                spx_open = float(_p)
                break
        logger.debug(f"Drift guard: SPX open = {spx_open}")

        guards = _DriftGuards(spx_open=spx_open)

        # Pre-scan 09:31 → entry_start: latch drift flags before entry window opens
        if spx_open and spx_open > 0:
            pre_times = [
                t for t in _build_minute_grid(date, "09:31:00", entry_start)
                if t < entry_start
            ]
            for _pt in pre_times:
                _pp = self._provider.get_fastest_spx_price(date, _pt)
                if _pp and _pp > 0:
                    self._step2_update_drift(float(_pp), guards, _pt)
            logger.debug(
                f"Pre-scan complete: put_blocked={guards.put_spread_ever_blocked} "
                f"call_blocked={guards.call_spread_ever_blocked} "
                f"ic_blocked={guards.ic_ever_blocked}"
            )

        # Open-position slots
        open_put_spread:  Optional[Any] = None
        open_call_spread: Optional[Any] = None
        open_ic:          Optional[Any] = None
        ic_leg_status:    Optional[IronCondorLegStatus] = None
        ic_entry_meta:    dict = {}
        put_spread_meta:  dict = {}
        call_spread_meta: dict = {}
        ic_checkpoints:         List[dict] = []
        put_spread_checkpoints: List[dict] = []
        call_spread_checkpoints: List[dict] = []

        # Day-level latches
        _had_call_spread_today     = False
        _had_call_spread_win_today = False
        _had_put_spread_win_today  = False

        # Per-bar SPX history for indicator computation (full mode only)
        _spx_history: List[float] = []

        trades: List[EnhancedBacktestResult] = []
        scan_times = _build_minute_grid(date, entry_start, FINAL_EXIT_TIME)

        for bar_index, current_time in enumerate(scan_times):
            is_past_entry_cutoff = current_time >= last_entry
            is_final_bar         = current_time >= FINAL_EXIT_TIME
            is_check_bar         = (bar_index % config.monitor_interval == 0) or is_final_bar

            # ── STEP 1: get SPX price ──────────────────────────────────
            # For real-time providers this also paces the loop to wall-clock time.
            spx = self._step1_get_spx_price(date, current_time)
            if spx is None:
                continue

            _spx_history.append(spx)

            # ── STEP 2: update drift guards ────────────────────────────
            self._step2_update_drift(spx, guards, current_time)

            # ── Monitor existing positions (check bars only) ───────────

            # --- IC monitoring ---
            if open_ic is not None and ic_leg_status is not None and is_check_bar:
                ic_leg_status = monitor.check_ic_leg_decay(
                    open_ic, date, current_time, ic_leg_status
                )

                # Collect checkpoint values for this bar
                ic_qty    = getattr(open_ic, 'quantity', 1)
                ic_credit = getattr(open_ic, 'entry_credit', 0)
                if is_final_bar:
                    raw_put  = self._engine._expiry_exit_cost(
                        open_ic, StrategyType.PUT_SPREAD, spx
                    )
                    raw_call = self._engine._expiry_exit_cost(
                        open_ic, StrategyType.CALL_SPREAD, spx
                    )
                    if not ic_leg_status.put_side_closed:
                        ic_leg_status.put_side_closed     = True
                        ic_leg_status.put_side_exit_time  = current_time
                        ic_leg_status.put_side_exit_cost  = raw_put
                        ic_leg_status.put_side_exit_reason = "Expired at market close"
                    if not ic_leg_status.call_side_closed:
                        ic_leg_status.call_side_closed     = True
                        ic_leg_status.call_side_exit_time  = current_time
                        ic_leg_status.call_side_exit_cost  = raw_call
                        ic_leg_status.call_side_exit_reason = "Expired at market close"
                else:
                    try:
                        _, _, raw_put, raw_call = monitor._check_ic_leg_decay_values(open_ic)
                    except Exception:
                        raw_put, raw_call = 0.0, 0.0

                ec_ps    = round(ic_credit / (100.0 * ic_qty), 4)
                put_ps   = round(raw_put   / (100.0 * ic_qty), 4)
                call_ps  = round(raw_call  / (100.0 * ic_qty), 4)
                total_ps = round(put_ps + call_ps, 4)
                ic_checkpoints.append({
                    "time": current_time, "spx": round(spx, 2),
                    "cost_per_share": total_ps,
                    "pnl_per_share":  round(ec_ps - total_ps, 4),
                    "put_cost_per_share":  put_ps,
                    "call_cost_per_share": call_ps,
                })
                fire({
                    "event": "monitor_tick",
                    "strategy_type": "Iron Condor",
                    "entry_time": ic_entry_meta.get("entry_time", ""),
                    "time": current_time, "spx": round(spx, 2),
                    "pnl_per_share": round(ec_ps - total_ps, 4),
                    "entry_credit_per_share": ec_ps,
                })
                logger.debug(
                    f"STEP 6 | {current_time} | IC | "
                    f"put_closed={ic_leg_status.put_side_closed} "
                    f"call_closed={ic_leg_status.call_side_closed} | "
                    f"cost={total_ps:.3f}/share | "
                    + ("EXIT: both sides" if (ic_leg_status.put_side_closed and ic_leg_status.call_side_closed) else "HOLD")
                )

                if ic_leg_status.put_side_closed and ic_leg_status.call_side_closed:
                    total_exit = (
                        ic_leg_status.put_side_exit_cost +
                        ic_leg_status.call_side_exit_cost
                    )
                    pnl = ic_credit - total_exit
                    later_time = max(
                        ic_leg_status.put_side_exit_time  or "00:00:00",
                        ic_leg_status.call_side_exit_time or "00:00:00",
                    )
                    result = self._make_result(
                        date, StrategyType.IRON_CONDOR, ic_entry_meta,
                        ic_credit, total_exit, pnl, spx, later_time,
                        "IC both sides closed", ic_checkpoints, ic_leg_status,
                    )
                    trades.append(result)
                    closed_ic   = open_ic
                    open_ic     = None
                    ic_leg_status   = None
                    ic_entry_meta   = {}
                    ic_checkpoints  = []
                    logger.info(
                        f"STEP 6 | {current_time} | IC EXIT | "
                        f"pnl={pnl:.2f} | reason=both-sides-closed"
                    )
                    order = self._broker.close_position(
                        closed_ic, quantity, later_time, total_exit
                    )
                    fire({
                        "event": "position_closed",
                        "strategy_type": "Iron Condor",
                        "result": result,
                        "strategy_obj": closed_ic,
                        "exit_time": later_time,
                        "order_result": order,
                    })

            # --- Put spread monitoring ---
            if open_put_spread is not None and is_check_bar:
                should_exit, current_cost, reason = monitor.check_decay_at_time(
                    open_put_spread, StrategyType.PUT_SPREAD, date, current_time
                )
                if is_final_bar:
                    current_cost = self._engine._expiry_exit_cost(
                        open_put_spread, StrategyType.PUT_SPREAD, spx
                    )
                    should_exit = True
                    reason = "Expired at market close"

                ps_qty    = getattr(open_put_spread, 'quantity', 1)
                ps_credit = getattr(open_put_spread, 'entry_credit', 0)
                ec_ps   = round(ps_credit   / (100.0 * ps_qty), 4)
                cost_ps = round(current_cost / (100.0 * ps_qty), 4)
                put_spread_checkpoints.append({
                    "time": current_time, "spx": round(spx, 2),
                    "cost_per_share": cost_ps,
                    "pnl_per_share":  round(ec_ps - cost_ps, 4),
                })
                fire({
                    "event": "monitor_tick",
                    "strategy_type": "Put Spread",
                    "entry_time": put_spread_meta.get("entry_time", current_time),
                    "time": current_time, "spx": round(spx, 2),
                    "pnl_per_share": round(ec_ps - cost_ps, 4),
                    "entry_credit_per_share": ec_ps,
                })
                logger.debug(
                    f"STEP 6 | {current_time} | put spread | "
                    f"cost={cost_ps:.3f}/share | "
                    + ("EXIT: " + reason if should_exit else "HOLD")
                )

                if should_exit:
                    pnl = ps_credit - current_cost
                    result = self._make_result(
                        date, StrategyType.PUT_SPREAD, put_spread_meta,
                        ps_credit, current_cost, pnl, spx, current_time,
                        reason, put_spread_checkpoints,
                    )
                    trades.append(result)
                    closed_ps           = open_put_spread
                    open_put_spread     = None
                    put_spread_meta     = {}
                    put_spread_checkpoints = []
                    if pnl > 0:
                        _had_put_spread_win_today = True
                    logger.info(
                        f"STEP 6 | {current_time} | PUT SPREAD EXIT | "
                        f"pnl={pnl:.2f} | reason={reason}"
                    )
                    order = self._broker.close_position(
                        closed_ps, quantity, current_time, current_cost
                    )
                    fire({
                        "event": "position_closed",
                        "strategy_type": "Put Spread",
                        "result": result,
                        "strategy_obj": closed_ps,
                        "exit_time": current_time,
                        "order_result": order,
                    })

            # --- Call spread monitoring ---
            if open_call_spread is not None and is_check_bar:
                should_exit, current_cost, reason = monitor.check_decay_at_time(
                    open_call_spread, StrategyType.CALL_SPREAD, date, current_time
                )
                if is_final_bar:
                    current_cost = self._engine._expiry_exit_cost(
                        open_call_spread, StrategyType.CALL_SPREAD, spx
                    )
                    should_exit = True
                    reason = "Expired at market close"

                cs_qty    = getattr(open_call_spread, 'quantity', 1)
                cs_credit = getattr(open_call_spread, 'entry_credit', 0)
                ec_ps   = round(cs_credit    / (100.0 * cs_qty), 4)
                cost_ps = round(current_cost / (100.0 * cs_qty), 4)
                call_spread_checkpoints.append({
                    "time": current_time, "spx": round(spx, 2),
                    "cost_per_share": cost_ps,
                    "pnl_per_share":  round(ec_ps - cost_ps, 4),
                })
                fire({
                    "event": "monitor_tick",
                    "strategy_type": "Call Spread",
                    "entry_time": call_spread_meta.get("entry_time", current_time),
                    "time": current_time, "spx": round(spx, 2),
                    "pnl_per_share": round(ec_ps - cost_ps, 4),
                    "entry_credit_per_share": ec_ps,
                })
                logger.debug(
                    f"STEP 6 | {current_time} | call spread | "
                    f"cost={cost_ps:.3f}/share | "
                    + ("EXIT: " + reason if should_exit else "HOLD")
                )

                if should_exit:
                    pnl = cs_credit - current_cost
                    result = self._make_result(
                        date, StrategyType.CALL_SPREAD, call_spread_meta,
                        cs_credit, current_cost, pnl, spx, current_time,
                        reason, call_spread_checkpoints,
                    )
                    trades.append(result)
                    closed_cs            = open_call_spread
                    open_call_spread     = None
                    call_spread_meta     = {}
                    call_spread_checkpoints = []
                    if pnl > 0:
                        _had_call_spread_win_today = True
                    logger.info(
                        f"STEP 6 | {current_time} | CALL SPREAD EXIT | "
                        f"pnl={pnl:.2f} | reason={reason}"
                    )
                    order = self._broker.close_position(
                        closed_cs, quantity, current_time, current_cost
                    )
                    fire({
                        "event": "position_closed",
                        "strategy_type": "Call Spread",
                        "result": result,
                        "strategy_obj": closed_cs,
                        "exit_time": current_time,
                        "order_result": order,
                    })

            # ── STEP 4: evaluate entry ─────────────────────────────────
            if not is_past_entry_cutoff and not is_final_bar:
                spx_series = pd.Series(_spx_history) if _spx_history else None

                strategy, s_type, meta = self._step4_evaluate_entry(
                    date, current_time, spx, spx_series, guards, config,
                    open_ic, open_put_spread, open_call_spread,
                    _had_call_spread_today,
                    _had_call_spread_win_today,
                    _had_put_spread_win_today,
                )

                if strategy is not None:
                    # ── STEP 5: submit order ───────────────────────────
                    entry_credit = getattr(strategy, 'entry_credit', 0)
                    order = self._step5_send_order(
                        strategy, quantity, current_time, entry_credit
                    )

                    entry_event = {
                        "event": "position_opened",
                        "strategy_type": s_type.value,
                        "entry_time": current_time,
                        "entry_spx": spx,
                        "entry_credit": entry_credit,
                        "strikes": meta.get("strike_selection"),
                        "entry_rationale": meta.get("entry_rationale"),
                        "strategy_obj": strategy,
                        "order_result": order,
                    }

                    if s_type == StrategyType.IRON_CONDOR:
                        open_ic       = strategy
                        ic_leg_status = IronCondorLegStatus()
                        ic_entry_meta = meta
                        logger.info(f"Opened IC at {current_time}")
                        fire(entry_event)

                    elif s_type == StrategyType.PUT_SPREAD:
                        open_put_spread = strategy
                        put_spread_meta = meta
                        logger.info(f"Opened put spread at {current_time}")
                        fire(entry_event)

                    elif s_type == StrategyType.CALL_SPREAD:
                        open_call_spread    = strategy
                        _had_call_spread_today = True
                        call_spread_meta    = meta
                        logger.info(f"Opened call spread at {current_time}")
                        fire(entry_event)

        # ── Force-close any remaining open positions at FINAL_EXIT_TIME ──
        final_spx = self._provider.get_fastest_spx_price(date, FINAL_EXIT_TIME) or 0.0
        for _open_pos, _meta, _stype, _checkpoints in [
            (open_ic,         ic_entry_meta,    StrategyType.IRON_CONDOR, ic_checkpoints),
            (open_put_spread,  put_spread_meta,  StrategyType.PUT_SPREAD,  put_spread_checkpoints),
            (open_call_spread, call_spread_meta, StrategyType.CALL_SPREAD, call_spread_checkpoints),
        ]:
            if _open_pos is None:
                continue

            exit_cost    = self._engine._expiry_exit_cost(_open_pos, _stype, final_spx)
            entry_credit = getattr(_open_pos, 'entry_credit', 0)
            pnl          = entry_credit - exit_cost

            pos_qty = getattr(_open_pos, 'quantity', 1)
            ec_ps   = round(entry_credit / (100.0 * pos_qty), 4)
            cost_ps = round(exit_cost    / (100.0 * pos_qty), 4)
            final_cp: dict = {
                "time": FINAL_EXIT_TIME, "spx": round(final_spx, 2),
                "cost_per_share": cost_ps,
                "pnl_per_share":  round(ec_ps - cost_ps, 4),
            }
            if _stype == StrategyType.IRON_CONDOR:
                put_cost  = self._engine._expiry_exit_cost(
                    _open_pos, StrategyType.PUT_SPREAD,  final_spx
                )
                call_cost = self._engine._expiry_exit_cost(
                    _open_pos, StrategyType.CALL_SPREAD, final_spx
                )
                final_cp["put_cost_per_share"]  = round(put_cost  / (100.0 * pos_qty), 4)
                final_cp["call_cost_per_share"] = round(call_cost / (100.0 * pos_qty), 4)
            _checkpoints.append(final_cp)

            # Finalise IC leg status for expiry
            _ic_ls = None
            if _stype == StrategyType.IRON_CONDOR and ic_leg_status is not None:
                if not ic_leg_status.put_side_closed:
                    ic_leg_status.put_side_closed      = True
                    ic_leg_status.put_side_exit_time   = FINAL_EXIT_TIME
                    ic_leg_status.put_side_exit_reason = "Expired at market close"
                if not ic_leg_status.call_side_closed:
                    ic_leg_status.call_side_closed      = True
                    ic_leg_status.call_side_exit_time   = FINAL_EXIT_TIME
                    ic_leg_status.call_side_exit_reason = "Expired at market close"
                _ic_ls = ic_leg_status

            result = self._make_result(
                date, _stype, _meta, entry_credit, exit_cost, pnl,
                final_spx, FINAL_EXIT_TIME, "Expired at market close",
                _checkpoints, _ic_ls,
            )
            trades.append(result)
            logger.info(
                f"STEP 6 | {FINAL_EXIT_TIME} | {_stype.value} EXPIRY EXIT | "
                f"pnl={pnl:.2f}"
            )
            order = self._broker.close_position(
                _open_pos, quantity, FINAL_EXIT_TIME, exit_cost
            )
            fire({
                "event": "position_closed",
                "strategy_type": _stype.value,
                "result": result,
                "strategy_obj": _open_pos,
                "exit_time": FINAL_EXIT_TIME,
                "order_result": order,
            })

        return DayBacktestResult(
            date=date,
            trades=trades,
            total_pnl=sum(t.pnl for t in trades),
            trade_count=len(trades),
            scan_minutes_checked=len(scan_times),
        )

    # ------------------------------------------------------------------
    # Step 1: get SPX price
    # ------------------------------------------------------------------

    def _step1_get_spx_price(self, date: str, bar_time: str) -> Optional[float]:
        """
        Fetch live SPX price for this bar.  For RealtimeMarketDataProvider
        this call blocks until the bar minute has elapsed (pacing the loop).
        """
        price = self._provider.get_fastest_spx_price(date, bar_time)
        if price and price > 0:
            logger.info(f"STEP 1 | {bar_time} | SPX={price:.2f}")
            return float(price)
        logger.debug(f"STEP 1 | {bar_time} | SPX=N/A")
        return None

    # ------------------------------------------------------------------
    # Step 2: update drift guards
    # ------------------------------------------------------------------

    def _step2_update_drift(
        self, spx: float, guards: _DriftGuards, bar_time: str
    ) -> float:
        """
        Update intraday drift extremes and latch drift block flags.
        Returns current drift from open (points).
        """
        if guards.spx_open is None and spx > 0:
            guards.spx_open = spx
            logger.debug(f"Drift guard: latched open = {spx:.2f} at {bar_time}")

        drift = (spx - guards.spx_open) if guards.spx_open else 0.0
        guards.intraday_max_drift = max(guards.intraday_max_drift, drift)
        guards.intraday_min_drift = min(guards.intraday_min_drift, drift)

        if drift >= DRIFT_BLOCK_POINTS:
            guards.put_spread_ever_blocked = True
        if drift <= -DRIFT_BLOCK_POINTS:
            if not guards.call_spread_ever_blocked:
                guards.call_blocked_latch_time = bar_time
            guards.call_spread_ever_blocked = True
        if abs(drift) >= DRIFT_IC_BLOCK_POINTS:
            guards.ic_ever_blocked = True

        logger.debug(
            f"STEP 2 | {bar_time} | drift={drift:+.1f} "
            f"put_blocked={guards.put_spread_ever_blocked} "
            f"call_blocked={guards.call_spread_ever_blocked} "
            f"ic_blocked={guards.ic_ever_blocked}"
        )
        return drift

    # ------------------------------------------------------------------
    # Step 4: evaluate entry
    # ------------------------------------------------------------------

    def _step4_evaluate_entry(
        self,
        date: str,
        bar_time: str,
        spx: float,
        spx_series: Optional[pd.Series],
        guards: _DriftGuards,
        config,
        open_ic,
        open_put_spread,
        open_call_spread,
        had_call_spread_today: bool,
        had_call_spread_win_today: bool,
        had_put_spread_win_today: bool,
    ) -> Tuple[Optional[Any], Optional[StrategyType], dict]:
        """
        Evaluate whether to open a new position at this bar.
        Returns (strategy, strategy_type, entry_meta) or (None, None, {}).
        """
        if spx <= 0:
            return None, None, {}

        strategy_mode = config.strategy.value
        skip_ind      = config.skip_indicators
        day_drift     = (spx - guards.spx_open) if guards.spx_open else 0.0

        # Apply sticky drift block flags (already latched by _step2_update_drift)
        ic_blocked   = guards.ic_ever_blocked
        put_blocked  = guards.put_spread_ever_blocked
        call_blocked = guards.call_spread_ever_blocked

        # Intraday reversal guards
        if (guards.intraday_max_drift - day_drift) >= INTRADAY_CALL_REVERSAL_POINTS:
            call_blocked = True
        if (day_drift - guards.intraday_min_drift) >= INTRADAY_PUT_REVERSAL_POINTS:
            put_blocked = True

        # Re-entry guards
        if had_call_spread_today and day_drift < 0:
            call_blocked = True
        if had_call_spread_today and day_drift > 15:
            call_blocked = True

        # Determine allowed types based on strategy mode
        allow_ic      = (
            strategy_mode in (STRATEGY_IRON_CONDOR, STRATEGY_IC_CREDIT_SPREADS)
            and not ic_blocked
        )
        allow_spreads = strategy_mode in (STRATEGY_CREDIT_SPREADS, STRATEGY_IC_CREDIT_SPREADS)

        # Determine direction from drift (simple mode default)
        if strategy_mode in (STRATEGY_IRON_CONDOR, STRATEGY_IC_CREDIT_SPREADS):
            sel_type = StrategyType.IRON_CONDOR
        elif day_drift < -5:
            sel_type = StrategyType.CALL_SPREAD   # market fell → sell above
        else:
            sel_type = StrategyType.PUT_SPREAD     # flat/up → sell below

        # Neutral indicator sentinel (overwritten in full mode)
        indicators = TechnicalIndicators(
            rsi=50.0, macd_line=0.0, macd_signal=0.0, macd_histogram=0.0,
            bb_upper=spx, bb_middle=spx, bb_lower=spx, bb_position=0.5,
        )

        if not skip_ind and spx_series is not None and len(spx_series) >= 15:
            indicators = TechnicalAnalyzer.analyze_market_conditions(spx_series)
            if indicators.rsi == 50.0 and indicators.bb_position == 0.5:
                logger.info(
                    f"STEP 4 | {bar_time} | warming up ({len(spx_series)} bars)"
                )
                return None, None, {}

            # Full-mode RSI / BB guards
            if indicators.bb_position > 0.94:
                call_blocked = True
            if indicators.bb_position < 0.05:
                put_blocked = True
            if day_drift < 0 and day_drift > -50 and indicators.rsi < 30:
                put_blocked = True
            if indicators.rsi < 12:
                put_blocked = True
            if day_drift < -50 and indicators.rsi < 20:
                put_blocked = True
            if indicators.rsi > CALL_SPREAD_MAX_ENTRY_RSI:
                call_blocked = True
            if indicators.bb_position > 0.90 and day_drift < -5:
                call_blocked = True
            if indicators.rsi < 40 and day_drift < -5:
                call_blocked = True
            if indicators.rsi < 30:
                call_blocked = True
            if day_drift > 12 and indicators.rsi < 40:
                call_blocked = True
            if indicators.bb_position < 0.10:
                call_blocked = True
            if day_drift < -3 and indicators.bb_position > 0.82:
                call_blocked = True
            if indicators.rsi < 32:
                call_blocked = True
            if had_call_spread_win_today and indicators.rsi > 20:
                put_blocked = True
            if had_put_spread_win_today and indicators.rsi < 40 and indicators.bb_position < 0.50:
                call_blocked = True
            if open_call_spread is not None and indicators.rsi < 40:
                put_blocked = True
            if open_put_spread is not None and indicators.rsi < 40 and indicators.bb_position < 0.50:
                call_blocked = True
            if open_put_spread is not None and indicators.rsi > 65:
                call_blocked = True
            if day_drift < 0 and indicators.rsi > 74:
                call_blocked = True

            # Use strategy selector for signal when running full indicators
            sel_type = self._engine.strategy_selector.select_strategy(indicators).strategy_type

        # Guard summary for logging
        guard_parts = []
        if ic_blocked:    guard_parts.append("IC:drift-extreme")
        if put_blocked:   guard_parts.append("put:BLOCKED")
        if call_blocked:  guard_parts.append("call:BLOCKED")
        if not guard_parts: guard_parts.append("all-clear")

        logger.info(
            f"STEP 4 | {bar_time} | signal={sel_type.value} "
            f"drift={day_drift:+.1f} | guards={' '.join(guard_parts)}"
        )

        # Dispatch to correct strategy type
        try:
            if sel_type == StrategyType.IRON_CONDOR and allow_ic:
                if open_ic is None and open_put_spread is None and open_call_spread is None:
                    return self._try_build_strategy(
                        date, bar_time, StrategyType.IRON_CONDOR,
                        config, spx, day_drift, guards, indicators,
                        ic_blocked, put_blocked, call_blocked,
                    )

            elif sel_type == StrategyType.PUT_SPREAD and allow_spreads and not put_blocked:
                put_ok = self._check_put_ok(
                    skip_ind, day_drift, indicators, call_blocked,
                    guards, date, bar_time
                )
                if open_put_spread is None and open_ic is None and put_ok:
                    return self._try_build_strategy(
                        date, bar_time, StrategyType.PUT_SPREAD,
                        config, spx, day_drift, guards, indicators,
                        ic_blocked, put_blocked, call_blocked,
                    )

            elif sel_type == StrategyType.CALL_SPREAD and allow_spreads and not call_blocked:
                if open_call_spread is None and open_ic is None:
                    return self._try_build_strategy(
                        date, bar_time, StrategyType.CALL_SPREAD,
                        config, spx, day_drift, guards, indicators,
                        ic_blocked, put_blocked, call_blocked,
                    )

        except Exception as exc:
            logger.debug(f"STEP 4 | {bar_time} | entry error: {exc}")

        return None, None, {}

    def _check_put_ok(
        self,
        skip_ind: bool,
        day_drift: float,
        indicators: TechnicalIndicators,
        call_blocked: bool,
        guards: _DriftGuards,
        date: str,
        bar_time: str,
    ) -> bool:
        """
        Apply RSI/cooldown conviction checks for put spread entry.
        In simple mode this always returns True (drift guards already applied).
        """
        if skip_ind:
            return True
        if day_drift < 0:
            rsi_ok = indicators.rsi <= PUT_SPREAD_MAX_RSI_ON_NEG_DRIFT
            if call_blocked and guards.call_blocked_latch_time:
                latch_dt   = pd.Timestamp(f"{date} {guards.call_blocked_latch_time}")
                current_dt = pd.Timestamp(f"{date} {bar_time}")
                cooldown_ok = (current_dt - latch_dt).seconds / 60 >= PUT_SPREAD_DRIFT_CONFIRM_MINUTES
            else:
                cooldown_ok = False
            return call_blocked and rsi_ok and cooldown_ok
        # Positive / flat drift: require strong oversold reading
        return indicators.rsi <= PUT_SPREAD_MAX_RSI_ON_POS_DRIFT

    def _try_build_strategy(
        self,
        date: str,
        bar_time: str,
        s_type: StrategyType,
        config,
        spx: float,
        day_drift: float,
        guards: _DriftGuards,
        indicators: TechnicalIndicators,
        ic_blocked: bool,
        put_blocked: bool,
        call_blocked: bool,
    ) -> Tuple[Optional[Any], Optional[StrategyType], dict]:
        """
        Select strikes and build a strategy object.
        Returns (strategy, strategy_type, entry_meta) or (None, None, {}).
        """
        # Strike selection (uses self._delta_selector with live provider)
        selection = self._delta_selector.select_strikes_by_delta(
            date=date,
            timestamp=bar_time,
            strategy_type=s_type,
            target_credit=config.target_credit,
            min_spread_width=config.spread_width,
        )
        if selection is None:
            logger.info(f"STEP 4 | {bar_time} | No strikes found for {s_type.value}")
            return None, None, {}

        # Minimum distance guard
        base_dist = MIN_DISTANCE_IC if s_type == StrategyType.IRON_CONDOR else MIN_DISTANCE_SPREAD
        if isinstance(selection, IronCondorStrikeSelection):
            put_dist  = abs(selection.put_short_strike  - spx)
            call_dist = abs(selection.call_short_strike - spx)
            if put_dist < base_dist or call_dist < base_dist:
                logger.info(
                    f"STEP 4 | BLOCKED: IC min-distance "
                    f"put={put_dist:.1f} call={call_dist:.1f} (min={base_dist})"
                )
                return None, None, {}
            if put_dist < config.spread_width or call_dist < config.spread_width:
                logger.info(f"STEP 4 | BLOCKED: IC circuit-breaker")
                return None, None, {}
        else:
            short_dist = abs(selection.short_strike - spx)
            if short_dist < base_dist:
                logger.info(
                    f"STEP 4 | BLOCKED: {s_type.value} min-distance "
                    f"short_dist={short_dist:.1f} (min={base_dist})"
                )
                return None, None, {}
            if short_dist < config.spread_width:
                logger.info(f"STEP 4 | BLOCKED: {s_type.value} circuit-breaker")
                return None, None, {}

        # Build strategy object (uses engine._build_*_strategy which needs
        # enhanced_query_engine → swapped to live provider in run_day)
        if s_type == StrategyType.IRON_CONDOR:
            strategy = self._engine._build_iron_condor_strategy(
                date, bar_time, selection, config.contracts
            )
        elif s_type == StrategyType.PUT_SPREAD:
            strategy = self._engine._build_put_spread_strategy(
                date, bar_time, selection, config.contracts
            )
        else:
            strategy = self._engine._build_call_spread_strategy(
                date, bar_time, selection, config.contracts
            )

        if strategy is None:
            logger.info(f"STEP 4 | {bar_time} | Strategy build failed for {s_type.value}")
            return None, None, {}

        entry_credit = getattr(strategy, 'entry_credit', 0)
        logger.info(
            f"STEP 4 | ENTER {s_type.value} | {bar_time} | "
            f"credit={entry_credit:.2f} drift={day_drift:+.1f}"
        )

        meta = {
            "entry_time":       bar_time,
            "entry_spx":        spx,
            "strike_selection": selection,
            "market_signal":    MarketSignal.NEUTRAL,
            "confidence":       0.5,
            "notes":            f"IBKR live drift={day_drift:+.1f}",
            "entry_rationale": {
                "strategy_selected":  s_type.value,
                "spx_at_entry":       round(spx, 2),
                "day_drift_pts":      round(day_drift, 2),
                "ic_blocked_by_drift":  ic_blocked,
                "put_spread_blocked":   put_blocked,
                "call_spread_blocked":  call_blocked,
                "rsi":         round(indicators.rsi, 2),
                "bb_position": round(indicators.bb_position, 3),
            },
        }
        return strategy, s_type, meta

    # ------------------------------------------------------------------
    # Step 5: send order
    # ------------------------------------------------------------------

    def _step5_send_order(
        self, strategy, quantity: int, bar_time: str, credit: float
    ) -> OrderResult:
        """Submit open-position order to the broker."""
        order = self._broker.open_position(strategy, quantity, bar_time, credit)
        if order.success:
            logger.info(
                f"STEP 5 | FILLED @ {order.fill_price:.2f} "
                f"(target {order.limit_price:.2f} slippage {order.slippage:+.2f}) "
                f"| {bar_time}"
            )
        else:
            logger.warning(
                f"STEP 5 | NO FILL | {bar_time} | {order.error_message}"
            )
        return order

    # ------------------------------------------------------------------
    # Helper: build EnhancedBacktestResult
    # ------------------------------------------------------------------

    def _make_result(
        self,
        date: str,
        stype: StrategyType,
        meta: dict,
        entry_credit: float,
        exit_cost: float,
        pnl: float,
        exit_spx: float,
        exit_time: str,
        exit_reason: str,
        checkpoints: list,
        ic_leg_status=None,
    ) -> EnhancedBacktestResult:
        pnl_pct = (pnl / entry_credit * 100) if entry_credit else 0.0
        entry_spx = meta.get("entry_spx", 0)
        return EnhancedBacktestResult(
            date=date,
            strategy_type=stype,
            market_signal=meta.get("market_signal", MarketSignal.NEUTRAL),
            entry_time=meta.get("entry_time", exit_time),
            exit_time=exit_time,
            exit_reason=exit_reason,
            entry_spx_price=entry_spx,
            exit_spx_price=exit_spx,
            technical_indicators=meta.get(
                "indicators",
                TechnicalIndicators(0, 0, 0, 0, 0, 0, 0, 0.5)
            ),
            strike_selection=meta.get("strike_selection", StrikeSelection(0, 0, 0, 0, 0)),
            entry_credit=entry_credit,
            exit_cost=exit_cost,
            pnl=pnl,
            pnl_pct=pnl_pct,
            max_profit=entry_credit,
            max_loss=-exit_cost,
            monitoring_points=checkpoints,
            success=True,
            confidence=meta.get("confidence", 0),
            notes=meta.get("notes", ""),
            entry_rationale=meta.get("entry_rationale"),
            exit_rationale={
                "exit_trigger": exit_reason,
                "exit_cost":    round(exit_cost, 2),
                "entry_credit": round(entry_credit, 2),
                "spx_at_exit":  round(exit_spx, 2),
                "spx_move_since_entry": round(exit_spx - entry_spx, 2) if entry_spx else None,
                "pnl":     round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
            },
            ic_leg_status=ic_leg_status,
        )
