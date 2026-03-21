"""Finance-oriented tool functions for QuantLab TDLC workflows."""

from __future__ import annotations

import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Annotated, Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import requests

from functions.function_calling.file import FileToolContext
from utils.function_catalog import ParamMeta

_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_REQUEST_TIMEOUT = 20
_DEFAULT_THRESHOLDS = {
    "min_sharpe": 1.0,
    "min_sortino": 1.0,
    "max_drawdown": 0.20,
    "min_total_return": 0.0,
}
_DEFAULT_AGENT_WEIGHTS = {
    "Alpha Researcher": 0.18,
    "Portfolio Manager": 0.22,
    "Quant Architect": 0.24,
    "Algo Developer": 0.22,
    "Risk Validator": 0.14,
}


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _coerce_json_object(raw: Mapping[str, Any] | str | None, *, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if raw is None:
        return dict(default or {})
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return dict(default or {})
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object")
        return parsed
    raise ValueError("Expected a mapping or JSON object string")


def _coerce_json_list(raw: Sequence[Mapping[str, Any]] | str | None) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON array")
        return [dict(item) for item in parsed]
    return [dict(item) for item in raw]


def _resolve_workspace_path(path_value: str, _context: Dict[str, Any] | None = None) -> Path:
    raw_path = Path(path_value)
    if raw_path.is_absolute():
      return raw_path.resolve()
    if _context is None:
        return raw_path.resolve()
    ctx = FileToolContext(_context)
    return ctx.resolve_under_workspace(raw_path)


def _save_dataframe(df: pd.DataFrame, output_path: str | None, _context: Dict[str, Any] | None) -> str | None:
    if not output_path:
        return None
    target = _resolve_workspace_path(output_path, _context)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return str(target)


def _save_json(payload: Dict[str, Any], output_path: str | None, _context: Dict[str, Any] | None) -> str | None:
    if not output_path:
        return None
    target = _resolve_workspace_path(output_path, _context)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_json_dump(payload), encoding="utf-8")
    return str(target)


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "Date" in working.columns:
        working["Date"] = pd.to_datetime(working["Date"], utc=True, errors="coerce")
    else:
        first_column = working.columns[0]
        working[first_column] = pd.to_datetime(working[first_column], utc=True, errors="coerce")
        working = working.rename(columns={first_column: "Date"})
    working = working.dropna(subset=["Date"])
    return working.sort_values("Date").reset_index(drop=True)


def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = _parse_dates(df)
    rename_map = {}
    for column in working.columns:
        lowered = column.strip().lower()
        if lowered == "adj close":
            rename_map[column] = "AdjClose"
        elif lowered == "close":
            rename_map[column] = "Close"
        elif lowered == "open":
            rename_map[column] = "Open"
        elif lowered == "high":
            rename_map[column] = "High"
        elif lowered == "low":
            rename_map[column] = "Low"
        elif lowered == "volume":
            rename_map[column] = "Volume"
        elif lowered == "signal":
            rename_map[column] = "signal"
        elif lowered == "prediction":
            rename_map[column] = "prediction"
    working = working.rename(columns=rename_map)
    if "AdjClose" not in working.columns and "Close" in working.columns:
        working["AdjClose"] = working["Close"]
    required = ["Date", "AdjClose"]
    missing = [name for name in required if name not in working.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for numeric in ["AdjClose", "Close", "Open", "High", "Low", "Volume", "signal", "prediction"]:
        if numeric in working.columns:
            working[numeric] = pd.to_numeric(working[numeric], errors="coerce")
    working = working.dropna(subset=["AdjClose"])
    return working.reset_index(drop=True)


def _fetch_yahoo_history(
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    interval: str,
) -> pd.DataFrame:
    params: Dict[str, Any] = {"interval": interval, "includePrePost": "false", "events": "div,splits"}
    if start_date and end_date:
        params["period1"] = int(pd.Timestamp(start_date, tz="UTC").timestamp())
        params["period2"] = int(pd.Timestamp(end_date, tz="UTC").timestamp())
    else:
        params["range"] = "1y"

    response = requests.get(
        _YAHOO_CHART_URL.format(symbol=symbol),
        params=params,
        timeout=_REQUEST_TIMEOUT,
        headers={"User-Agent": "QuantLab-TDLC/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo Finance error for {symbol}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise ValueError(f"No chart data returned for {symbol}")

    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [None])[0] or {}
    adjclose_series = ((result.get("indicators") or {}).get("adjclose") or [None])[0] or {}
    adjclose = adjclose_series.get("adjclose") or quote.get("close") or []
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s", utc=True),
            "Open": quote.get("open") or [],
            "High": quote.get("high") or [],
            "Low": quote.get("low") or [],
            "Close": quote.get("close") or [],
            "AdjClose": adjclose,
            "Volume": quote.get("volume") or [],
        }
    )
    return _normalize_price_frame(frame)


def _periods_per_year(interval: str) -> int:
    mapping = {
        "1d": 252,
        "1wk": 52,
        "1mo": 12,
        "1h": 24 * 252,
    }
    return mapping.get(interval, 252)


def _build_positions(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    strategy_type = str(config.get("type", "moving_average_crossover")).strip().lower()
    if strategy_type == "buy_and_hold":
        raw_signal = pd.Series(1.0, index=df.index)
    elif strategy_type == "moving_average_crossover":
        short_window = int(config.get("short_window", 20))
        long_window = int(config.get("long_window", 50))
        if short_window <= 0 or long_window <= 0 or short_window >= long_window:
            raise ValueError("moving_average_crossover requires 0 < short_window < long_window")
        short_ma = df["AdjClose"].rolling(window=short_window, min_periods=short_window).mean()
        long_ma = df["AdjClose"].rolling(window=long_window, min_periods=long_window).mean()
        raw_signal = (short_ma > long_ma).astype(float)
    elif strategy_type == "external_signal":
        signal_column = str(config.get("signal_column", "signal"))
        if signal_column not in df.columns:
            raise ValueError(f"external_signal requires column '{signal_column}'")
        raw_signal = pd.to_numeric(df[signal_column], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    elif strategy_type == "threshold_signal":
        prediction_column = str(config.get("prediction_column", "prediction"))
        upper = float(config.get("upper_threshold", 0.5))
        lower = float(config.get("lower_threshold", -0.5))
        if prediction_column not in df.columns:
            raise ValueError(f"threshold_signal requires column '{prediction_column}'")
        scores = pd.to_numeric(df[prediction_column], errors="coerce").fillna(0.0)
        raw_signal = pd.Series(0.0, index=df.index)
        raw_signal = raw_signal.mask(scores >= upper, 1.0)
        raw_signal = raw_signal.mask(scores <= lower, -1.0)
    else:
        raise ValueError(f"Unsupported strategy type: {strategy_type}")
    return raw_signal.shift(1).fillna(0.0)


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve.div(running_max).sub(1.0)
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _annualized_return(total_return: float, observations: int, periods_per_year: int) -> float:
    if observations <= 0:
        return 0.0
    final_multiple = 1.0 + total_return
    if final_multiple <= 0:
        return -1.0
    return float(final_multiple ** (periods_per_year / observations) - 1.0)


def _annualized_ratio(returns: pd.Series, downside_only: bool, periods_per_year: int) -> float | None:
    if returns.empty:
        return None
    sample = returns[returns < 0] if downside_only else returns
    denominator = float(sample.std(ddof=0))
    if denominator <= 0:
        return None
    numerator = float(returns.mean()) * math.sqrt(periods_per_year)
    return numerator / denominator


def _status_from_metrics(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> tuple[str, List[str]]:
    reasons: List[str] = []
    sharpe = metrics.get("sharpe_ratio")
    sortino = metrics.get("sortino_ratio")
    max_drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
    total_return = float(metrics.get("total_return", 0.0))
    if sharpe is None or sharpe < float(thresholds.get("min_sharpe", 1.0)):
        reasons.append("Sharpe ratio below target")
    if sortino is None or sortino < float(thresholds.get("min_sortino", 1.0)):
        reasons.append("Sortino ratio below target")
    if max_drawdown > float(thresholds.get("max_drawdown", 0.20)):
        reasons.append("Max drawdown above limit")
    if total_return < float(thresholds.get("min_total_return", 0.0)):
        reasons.append("Total return below target")
    return ("PASS" if not reasons else "FAIL", reasons)


def search_market_intelligence(
    query: Annotated[str, ParamMeta(description="Market or macro query to search")],
    max_results: Annotated[int, ParamMeta(description="Maximum number of search hits to return")] = 5,
    region: Annotated[str, ParamMeta(description="DDGS region code, for example wt-wt or us-en")] = "wt-wt",
) -> str:
    """Search market intelligence sources using DuckDuckGo results."""

    from ddgs import DDGS

    if max_results <= 0:
        raise ValueError("max_results must be positive")

    results: List[Dict[str, Any]] = []
    with DDGS() as client:
        for item in client.text(query, region=region, max_results=max_results):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
            )

    return _json_dump({"query": query, "region": region, "results": results})


def search_arxiv_papers(
    query: Annotated[str, ParamMeta(description="ArXiv search terms")],
    max_results: Annotated[int, ParamMeta(description="Maximum number of papers to return")] = 5,
) -> str:
    """Search ArXiv for quantitative research papers."""

    if max_results <= 0:
        raise ValueError("max_results must be positive")
    response = requests.get(
        _ARXIV_API_URL,
        params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
        timeout=_REQUEST_TIMEOUT,
        headers={"User-Agent": "QuantLab-TDLC/1.0"},
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        authors = [author.findtext("atom:name", default="", namespaces=ns) for author in entry.findall("atom:author", ns)]
        papers.append(
            {
                "title": (entry.findtext("atom:title", default="", namespaces=ns) or "").strip(),
                "summary": " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split()),
                "published": entry.findtext("atom:published", default="", namespaces=ns),
                "url": entry.findtext("atom:id", default="", namespaces=ns),
                "authors": authors,
            }
        )
    return _json_dump({"query": query, "papers": papers})


def fetch_price_history(
    symbol: Annotated[str, ParamMeta(description="Ticker symbol, for example SPY or AAPL")],
    start_date: Annotated[str | None, ParamMeta(description="Inclusive start date in YYYY-MM-DD format")] = None,
    end_date: Annotated[str | None, ParamMeta(description="Inclusive end date in YYYY-MM-DD format")] = None,
    interval: Annotated[str, ParamMeta(description="Bar interval supported by Yahoo chart API")] = "1d",
    output_path: Annotated[str | None, ParamMeta(description="Optional CSV path to save the fetched data")] = None,
    _context: Dict[str, Any] | None = None,
) -> str:
    """Fetch OHLCV price history from Yahoo Finance and optionally save it as CSV."""

    frame = _fetch_yahoo_history(symbol=symbol, start_date=start_date, end_date=end_date, interval=interval)
    saved_path = _save_dataframe(frame, output_path, _context)
    payload = {
        "symbol": symbol,
        "start_date": str(frame["Date"].iloc[0].date()) if not frame.empty else None,
        "end_date": str(frame["Date"].iloc[-1].date()) if not frame.empty else None,
        "rows": int(len(frame)),
        "interval": interval,
        "saved_path": saved_path,
        "preview": frame.head(5).assign(Date=lambda data: data["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
    }
    return _json_dump(payload)


def run_strategy_backtest(
    symbol: Annotated[str | None, ParamMeta(description="Ticker symbol used when price_csv is not provided")] = None,
    price_csv: Annotated[str | None, ParamMeta(description="CSV path containing Date and price columns, optionally signal or prediction columns")] = None,
    start_date: Annotated[str | None, ParamMeta(description="Inclusive start date in YYYY-MM-DD format")] = None,
    end_date: Annotated[str | None, ParamMeta(description="Inclusive end date in YYYY-MM-DD format")] = None,
    interval: Annotated[str, ParamMeta(description="Sampling interval used to annualize metrics")] = "1d",
    strategy_config: Annotated[str | Dict[str, Any] | None, ParamMeta(description="JSON object describing the strategy, for example {\"type\":\"moving_average_crossover\",\"short_window\":20,\"long_window\":50}")] = None,
    thresholds: Annotated[str | Dict[str, Any] | None, ParamMeta(description="JSON object of pass/fail thresholds, for example {\"min_sharpe\":1.0,\"max_drawdown\":0.2}")] = None,
    initial_capital: Annotated[float, ParamMeta(description="Initial capital used for the simulated equity curve")] = 100000.0,
    output_path: Annotated[str | None, ParamMeta(description="Optional JSON path to save the backtest report")] = None,
    _context: Dict[str, Any] | None = None,
) -> str:
    """Run a simple backtest and return financial metrics plus a PASS or FAIL decision."""

    if bool(symbol) == bool(price_csv):
        raise ValueError("Provide exactly one of symbol or price_csv")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    strategy = _coerce_json_object(strategy_config, default={"type": "moving_average_crossover", "short_window": 20, "long_window": 50})
    threshold_values = _coerce_json_object(thresholds, default=_DEFAULT_THRESHOLDS)

    if price_csv:
        csv_path = _resolve_workspace_path(price_csv, _context)
        frame = _normalize_price_frame(pd.read_csv(csv_path))
        data_source = str(csv_path)
        symbol_label = symbol or csv_path
    else:
        assert symbol is not None
        frame = _fetch_yahoo_history(symbol=symbol, start_date=start_date, end_date=end_date, interval=interval)
        data_source = f"yahoo:{symbol}"
        symbol_label = symbol

    if len(frame) < 3:
        raise ValueError("Backtest requires at least 3 observations")

    periods_per_year = _periods_per_year(interval)
    asset_returns = frame["AdjClose"].pct_change().fillna(0.0)
    position = _build_positions(frame, strategy)
    strategy_returns = asset_returns * position
    equity_curve = initial_capital * (1.0 + strategy_returns).cumprod()

    total_return = float(equity_curve.iloc[-1] / initial_capital - 1.0)
    net_profit = float(equity_curve.iloc[-1] - initial_capital)
    volatility = float(strategy_returns.std(ddof=0) * math.sqrt(periods_per_year)) if len(strategy_returns) else 0.0
    max_drawdown = _max_drawdown(equity_curve)
    annualized_return = _annualized_return(total_return, len(strategy_returns) - 1, periods_per_year)
    sharpe_ratio = _annualized_ratio(strategy_returns, False, periods_per_year)
    sortino_ratio = _annualized_ratio(strategy_returns, True, periods_per_year)
    trade_days = int((position != 0).sum())
    wins = int((strategy_returns > 0).sum())
    win_rate = float(wins / trade_days) if trade_days else 0.0
    exposure = float(position.abs().mean())
    turnover = float(position.diff().abs().fillna(0.0).sum())
    status, fail_reasons = _status_from_metrics(
        {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
        },
        threshold_values,
    )

    report = {
        "symbol": symbol_label,
        "data_source": str(data_source),
        "strategy": strategy,
        "thresholds": threshold_values,
        "status": status,
        "fail_reasons": fail_reasons,
        "metrics": {
            "observations": int(len(frame)),
            "start_date": str(frame["Date"].iloc[0].date()),
            "end_date": str(frame["Date"].iloc[-1].date()),
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "net_profit": net_profit,
            "win_rate": win_rate,
            "exposure": exposure,
            "turnover": turnover,
            "trade_days": trade_days,
            "ending_equity": float(equity_curve.iloc[-1]),
        },
        "bias_controls": {
            "position_shift_applied": True,
            "note": "Positions are shifted by one bar to avoid same-bar look-ahead bias in the simulator.",
        },
        "generated_at": int(time.time()),
    }
    report["saved_path"] = _save_json(report, output_path, _context)
    return _json_dump(report)


def evaluate_agent_kpis(
    agent_reports: Annotated[str | Sequence[Mapping[str, Any]], ParamMeta(description="JSON array of per-agent inputs with quality_score, decision_confidence, timeliness_score, and optional influence or risk_flags")],
    backtest_report: Annotated[str | Dict[str, Any], ParamMeta(description="Backtest report JSON returned by run_strategy_backtest")],
    owner_thresholds: Annotated[str | Dict[str, Any] | None, ParamMeta(description="Optional JSON object overriding KPI decision cutoffs")] = None,
    output_path: Annotated[str | None, ParamMeta(description="Optional JSON path to save the KPI report")] = None,
    _context: Dict[str, Any] | None = None,
) -> str:
    """Score each agent's contribution and recommend retain, suspend, or dismiss."""

    reports = _coerce_json_list(agent_reports)
    backtest_payload = _coerce_json_object(backtest_report)
    metrics = dict(backtest_payload.get("metrics") or {})
    thresholds = _coerce_json_object(owner_thresholds, default={
        "positive_profit_cutoff": 250.0,
        "negative_profit_cutoff": -250.0,
        "dismiss_score": 35.0,
        "suspend_score": 60.0,
    })
    if not reports:
        raise ValueError("agent_reports must contain at least one item")

    total_return = float(metrics.get("total_return", 0.0))
    sharpe = metrics.get("sharpe_ratio")
    sortino = metrics.get("sortino_ratio")
    max_drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
    net_profit = float(metrics.get("net_profit", 0.0))
    initial_capital = float(backtest_payload.get("strategy", {}).get("initial_capital", 100000.0))

    return_score = _clamp(total_return / max(abs(_DEFAULT_THRESHOLDS["min_total_return"]) + 0.05, 0.05), -1.0, 1.0)
    sharpe_score = _clamp(float(sharpe) / max(_DEFAULT_THRESHOLDS["min_sharpe"], 0.5), -1.0, 1.0) if sharpe is not None else -1.0
    sortino_score = _clamp(float(sortino) / max(_DEFAULT_THRESHOLDS["min_sortino"], 0.5), -1.0, 1.0) if sortino is not None else -1.0
    drawdown_score = _clamp(1.0 - (max_drawdown / max(_DEFAULT_THRESHOLDS["max_drawdown"], 0.05)), -1.0, 1.0)
    base_performance = 0.35 * return_score + 0.25 * sharpe_score + 0.20 * sortino_score + 0.20 * drawdown_score

    positive_cutoff = float(thresholds.get("positive_profit_cutoff", 250.0))
    negative_cutoff = float(thresholds.get("negative_profit_cutoff", -250.0))
    dismiss_score = float(thresholds.get("dismiss_score", 35.0))
    suspend_score = float(thresholds.get("suspend_score", 60.0))

    kpis: List[Dict[str, Any]] = []
    counts = {"retain": 0, "suspend": 0, "dismiss": 0}
    for report in reports:
        agent = str(report.get("agent") or report.get("name") or "Unknown Agent")
        phase = str(report.get("phase") or "unknown")
        quality_score = _clamp(float(report.get("quality_score", 0.5)), 0.0, 1.0)
        confidence = _clamp(float(report.get("decision_confidence", 0.5)), 0.0, 1.0)
        timeliness = _clamp(float(report.get("timeliness_score", 1.0)), 0.0, 1.0)
        error_penalty = _clamp(float(report.get("error_penalty", 0.0)), 0.0, 1.0)
        risk_flags = max(int(report.get("risk_flags", 0)), 0)
        influence = float(report.get("influence", _DEFAULT_AGENT_WEIGHTS.get(agent, 0.20)))
        influence = max(influence, 0.0)

        quality_component = (0.4 * quality_score) + (0.2 * confidence) + (0.2 * timeliness) + (0.2 * (1.0 - error_penalty))
        signed_quality = _clamp(((quality_component - 0.5) * 2.0) - min(risk_flags, 3) * 0.2, -1.0, 1.0)
        contribution_factor = _clamp((0.45 * base_performance) + (0.55 * signed_quality), -1.0, 1.0)
        attributed_profit = net_profit * influence * contribution_factor
        kpi_score = round(50.0 + (contribution_factor * 50.0), 2)

        if attributed_profit >= positive_cutoff:
            contribution_band = "positive"
            recommendation = "retain"
        elif attributed_profit <= negative_cutoff or kpi_score < dismiss_score:
            contribution_band = "negative"
            recommendation = "dismiss"
        elif kpi_score < suspend_score:
            contribution_band = "low"
            recommendation = "suspend"
        else:
            contribution_band = "low"
            recommendation = "suspend"

        counts[recommendation] = counts.get(recommendation, 0) + 1
        kpis.append(
            {
                "agent": agent,
                "phase": phase,
                "quality_score": quality_score,
                "decision_confidence": confidence,
                "timeliness_score": timeliness,
                "error_penalty": error_penalty,
                "risk_flags": risk_flags,
                "influence": influence,
                "kpi_score": kpi_score,
                "attributed_profit": round(attributed_profit, 2),
                "contribution_band": contribution_band,
                "recommended_action": recommendation,
            }
        )

    owner_summary = {
        "backtest_status": backtest_payload.get("status"),
        "net_profit": round(net_profit, 2),
        "base_performance_score": round(base_performance, 4),
        "decision_counts": counts,
        "owner_note": "Use retain or suspend as soft governance defaults and reserve dismiss for repeated negative contribution.",
    }
    result = {
        "owner_summary": owner_summary,
        "agent_kpis": kpis,
        "thresholds": thresholds,
        "initial_capital_reference": initial_capital,
    }
    result["saved_path"] = _save_json(result, output_path, _context)
    return _json_dump(result)