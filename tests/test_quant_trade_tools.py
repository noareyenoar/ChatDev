import json
from pathlib import Path

import pandas as pd

from functions.function_calling.quant_trade import evaluate_agent_kpis, run_strategy_backtest


def _write_price_csv(tmp_path: Path, *, with_signal: bool = False) -> Path:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "Close": [100, 101, 103, 104, 106, 108, 109, 111, 112, 115],
            "Adj Close": [100, 101, 103, 104, 106, 108, 109, 111, 112, 115],
        }
    )
    if with_signal:
        frame["signal"] = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    csv_path = tmp_path / "prices.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def test_run_strategy_backtest_buy_and_hold_returns_pass_metrics(tmp_path: Path):
    csv_path = _write_price_csv(tmp_path)
    payload = json.loads(
        run_strategy_backtest(
            price_csv=str(csv_path),
            strategy_config={"type": "buy_and_hold"},
            thresholds={"min_sharpe": 0.1, "min_sortino": 0.1, "max_drawdown": 0.5, "min_total_return": 0.0},
            initial_capital=100000.0,
        )
    )

    assert payload["status"] == "PASS"
    assert payload["metrics"]["net_profit"] > 0
    assert payload["metrics"]["total_return"] > 0
    assert payload["bias_controls"]["position_shift_applied"] is True


def test_run_strategy_backtest_external_signal_uses_csv_signal_column(tmp_path: Path):
    csv_path = _write_price_csv(tmp_path, with_signal=True)
    payload = json.loads(
        run_strategy_backtest(
            price_csv=str(csv_path),
            strategy_config={"type": "external_signal", "signal_column": "signal"},
            thresholds={"min_sharpe": 0.1, "min_sortino": 0.1, "max_drawdown": 0.5, "min_total_return": -0.1},
            initial_capital=100000.0,
        )
    )

    assert payload["metrics"]["trade_days"] > 0
    assert payload["metrics"]["ending_equity"] > 100000.0


def test_evaluate_agent_kpis_returns_owner_actions():
    backtest_report = {
        "status": "PASS",
        "strategy": {"initial_capital": 100000.0},
        "metrics": {
            "total_return": 0.12,
            "sharpe_ratio": 1.4,
            "sortino_ratio": 1.8,
            "max_drawdown": -0.08,
            "net_profit": 12000.0,
        },
    }
    agent_reports = [
        {
            "agent": "Quant Architect",
            "phase": "Model_Architecture",
            "quality_score": 0.95,
            "decision_confidence": 0.9,
            "timeliness_score": 1.0,
            "error_penalty": 0.0,
            "risk_flags": 0,
            "influence": 0.4,
        },
        {
            "agent": "Risk Validator",
            "phase": "Backtest_Execution",
            "quality_score": 0.2,
            "decision_confidence": 0.3,
            "timeliness_score": 0.4,
            "error_penalty": 0.8,
            "risk_flags": 3,
            "influence": 0.4,
        },
    ]

    payload = json.loads(evaluate_agent_kpis(agent_reports=agent_reports, backtest_report=backtest_report))
    actions = {item["agent"]: item["recommended_action"] for item in payload["agent_kpis"]}

    assert actions["Quant Architect"] == "retain"
    assert actions["Risk Validator"] in {"suspend", "dismiss"}
    assert payload["owner_summary"]["decision_counts"][actions["Quant Architect"]] >= 1