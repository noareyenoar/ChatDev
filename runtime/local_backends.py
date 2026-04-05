"""Local simulated backends for safe development and smoke testing.

Provides a ``SimulatedProvider`` that can drive simple workflows offline,
including the QuantLab simulated path, plus a tiny in-memory vector store.
"""

import json
import math
from typing import Any, List, Optional

from entity.messages import FunctionCallOutputEvent, Message, MessageRole, ToolCallPayload
from runtime.node.agent.providers.base import ModelProvider
from runtime.node.agent.providers.response import ModelResponse
from utils.token_tracker import TokenUsage


class SimulatedClient:
    """A tiny client placeholder for compatibility."""
    def __init__(self):
        pass


class SimulatedProvider(ModelProvider):
    """A deterministic, no-network provider for development/smoke runs."""

    def create_client(self) -> SimulatedClient:
        return SimulatedClient()

    def call_model(
        self,
        client: SimulatedClient,
        conversation: List[Message],
        timeline: List[Any],
        tool_specs: Optional[List[Any]] = None,
        **kwargs,
    ) -> ModelResponse:
        role_text = (self.config.role or "").strip()
        prompt_text = conversation[-1].text_content() if conversation else ""

        if "Alpha Researcher in QuantLab-TDLC" in role_text:
            return ModelResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=json.dumps(
                        {
                            "phase": "Alpha_Discovery",
                            "market_view": "Simulated bullish regime with persistent trend and manageable volatility.",
                            "tradable_universe": ["SPY"],
                            "hypotheses": [
                                {
                                    "name": "Trend persistence in broad equity index",
                                    "rationale": "A simple trend-following baseline is sufficient for offline regression and orchestration validation.",
                                    "evidence": [
                                        "Simulated market regime assumes positive drift",
                                        "Backtest tooling can verify position-shifted returns deterministically",
                                    ],
                                }
                            ],
                            "research_kpi": {
                                "agent": "Alpha Researcher",
                                "phase": "Alpha_Discovery",
                                "quality_score": 0.82,
                                "decision_confidence": 0.78,
                                "timeliness_score": 1.0,
                                "error_penalty": 0.0,
                                "risk_flags": 0,
                            },
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
                raw_response={"simulated": True, "node": "alpha_research"},
            )

        if "Portfolio Manager in QuantLab-TDLC" in role_text:
            return ModelResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=json.dumps(
                        {
                            "phase": "Alpha_Discovery",
                            "chosen_hypothesis": "Trend persistence in broad equity index",
                            "asset_universe": ["SPY"],
                            "risk_constraints": {
                                "max_drawdown": 0.25,
                                "gross_exposure_limit": 1.0,
                            },
                            "success_thresholds": {
                                "min_sharpe": 0.1,
                                "min_sortino": 0.1,
                                "min_total_return": 0.0,
                            },
                            "manager_kpi": {
                                "agent": "Portfolio Manager",
                                "phase": "Alpha_Discovery",
                                "quality_score": 0.8,
                                "decision_confidence": 0.8,
                                "timeliness_score": 1.0,
                                "error_penalty": 0.0,
                                "risk_flags": 0,
                            },
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
                raw_response={"simulated": True, "node": "portfolio_manager"},
            )

        if "Quant Architect in QuantLab-TDLC" in role_text:
            return ModelResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=json.dumps(
                        {
                            "phase": "Model_Architecture",
                            "model_family": "rules_based_trend_following",
                            "signal_definition": "Long when short moving average stays above long moving average or baseline buy-and-hold if deterministic smoke path is selected.",
                            "features": ["adj_close", "short_ma", "long_ma"],
                            "implementation_notes": [
                                "Persist a local CSV for offline replay.",
                                "Use buy_and_hold in the smoke path to avoid unnecessary model variance.",
                            ],
                            "targets": {
                                "min_sharpe": 0.1,
                                "min_sortino": 0.1,
                                "max_drawdown": 0.25,
                                "min_total_return": 0.0,
                            },
                            "architect_kpi": {
                                "agent": "Quant Architect",
                                "phase": "Model_Architecture",
                                "quality_score": 0.79,
                                "decision_confidence": 0.76,
                                "timeliness_score": 1.0,
                                "error_penalty": 0.0,
                                "risk_flags": 0,
                            },
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
                raw_response={"simulated": True, "node": "quant_architect"},
            )

        if "Algo Developer in QuantLab-TDLC" in role_text:
            if self._find_tool_event(timeline, "save_file") is None and self._has_tool(tool_specs, "save_file"):
                csv_payload = self._build_price_csv()
                return ModelResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content="Preparing a local deterministic price file for offline backtesting.",
                        tool_calls=[
                            ToolCallPayload(
                                id="sim-save-file",
                                function_name="save_file",
                                arguments=json.dumps(
                                    {
                                        "path": "quantlab/sim_prices.csv",
                                        "content": csv_payload,
                                        "mode": "overwrite",
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                        ],
                    ),
                    raw_response={"simulated": True, "node": "algo_developer", "step": "save_file"},
                )

            return ModelResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=json.dumps(
                        {
                            "phase": "Signal_Engineering",
                            "files_changed": ["quantlab/sim_prices.csv"],
                            "run_command": "uv run python run.py --path yaml_instance/QuantLab_TDLC_simulated.yaml --name quantlab_sim",
                            "backtest_contract": {
                                "price_csv": "quantlab/sim_prices.csv",
                                "strategy_config": {"type": "buy_and_hold"},
                            },
                            "developer_kpi": {
                                "agent": "Algo Developer",
                                "phase": "Signal_Engineering",
                                "quality_score": 0.84,
                                "decision_confidence": 0.81,
                                "timeliness_score": 1.0,
                                "error_penalty": 0.0,
                                "risk_flags": 0,
                            },
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
                raw_response={"simulated": True, "node": "algo_developer", "step": "final"},
            )

        if "Risk Validator in QuantLab-TDLC" in role_text:
            backtest_event = self._find_tool_event(timeline, "run_strategy_backtest")
            kpi_event = self._find_tool_event(timeline, "evaluate_agent_kpis")

            if backtest_event is None and self._has_tool(tool_specs, "run_strategy_backtest"):
                return ModelResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content="Running deterministic offline backtest.",
                        tool_calls=[
                            ToolCallPayload(
                                id="sim-backtest",
                                function_name="run_strategy_backtest",
                                arguments=json.dumps(
                                    {
                                        "price_csv": "quantlab/sim_prices.csv",
                                        "strategy_config": {"type": "buy_and_hold"},
                                        "thresholds": {
                                            "min_sharpe": 0.1,
                                            "min_sortino": 0.1,
                                            "max_drawdown": 0.5,
                                            "min_total_return": 0.0,
                                        },
                                        "initial_capital": 100000.0,
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                        ],
                    ),
                    raw_response={"simulated": True, "node": "risk_validator", "step": "backtest"},
                )

            if kpi_event is None and self._has_tool(tool_specs, "evaluate_agent_kpis") and backtest_event is not None:
                return ModelResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content="Scoring agent KPIs from the completed backtest.",
                        tool_calls=[
                            ToolCallPayload(
                                id="sim-kpis",
                                function_name="evaluate_agent_kpis",
                                arguments=json.dumps(
                                    {
                                        "agent_reports": self._default_agent_reports(),
                                        "backtest_report": backtest_event.output_text or "{}",
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                        ],
                    ),
                    raw_response={"simulated": True, "node": "risk_validator", "step": "kpis"},
                )

            backtest_payload = self._parse_json_payload(backtest_event.output_text if backtest_event else None)
            kpi_payload = self._parse_json_payload(kpi_event.output_text if kpi_event else None)
            decision = "APPROVED" if backtest_payload.get("status") == "PASS" else "REVISE_IMPLEMENTATION"

            return ModelResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=json.dumps(
                        {
                            "phase": "Recursive_Optimization",
                            "decision": decision,
                            "backtest_metrics": self._sanitize_json_value(backtest_payload.get("metrics", {})),
                            "fail_reasons": self._sanitize_json_value(backtest_payload.get("fail_reasons", [])),
                            "agent_kpis": self._sanitize_json_value(kpi_payload.get("agent_kpis", [])),
                            "owner_summary": self._sanitize_json_value(kpi_payload.get("owner_summary", {})),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
                raw_response={"simulated": True, "node": "risk_validator", "step": "final"},
            )

        reply_text = f"[SIMULATED] Reply to: {prompt_text[:200]}"
        return ModelResponse(
            message=Message(role=MessageRole.ASSISTANT, content=reply_text),
            raw_response={"simulated": True, "node": "generic"},
        )

    def extract_token_usage(self, response: Any) -> TokenUsage:
        text = ""
        if isinstance(response, dict):
            text = json.dumps(response, ensure_ascii=False)
        else:
            text = str(response)
        output_tokens = max(1, len(text) // 4) if text else 0
        return TokenUsage(input_tokens=0, output_tokens=output_tokens, total_tokens=output_tokens)

    def _has_tool(self, tool_specs: Optional[List[Any]], tool_name: str) -> bool:
        return any(getattr(spec, "name", None) == tool_name for spec in (tool_specs or []))

    def _find_tool_event(self, timeline: List[Any], tool_name: str) -> FunctionCallOutputEvent | None:
        for item in reversed(timeline):
            if isinstance(item, FunctionCallOutputEvent) and item.function_name == tool_name:
                return item
        return None

    def _parse_json_payload(self, raw: Optional[str]) -> dict:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _default_agent_reports(self) -> list[dict[str, Any]]:
        return [
            {
                "agent": "Alpha Researcher",
                "phase": "Alpha_Discovery",
                "quality_score": 0.82,
                "decision_confidence": 0.78,
                "timeliness_score": 1.0,
                "error_penalty": 0.0,
                "risk_flags": 0,
                "influence": 0.18,
            },
            {
                "agent": "Portfolio Manager",
                "phase": "Alpha_Discovery",
                "quality_score": 0.80,
                "decision_confidence": 0.80,
                "timeliness_score": 1.0,
                "error_penalty": 0.0,
                "risk_flags": 0,
                "influence": 0.22,
            },
            {
                "agent": "Quant Architect",
                "phase": "Model_Architecture",
                "quality_score": 0.79,
                "decision_confidence": 0.76,
                "timeliness_score": 1.0,
                "error_penalty": 0.0,
                "risk_flags": 0,
                "influence": 0.24,
            },
            {
                "agent": "Algo Developer",
                "phase": "Signal_Engineering",
                "quality_score": 0.84,
                "decision_confidence": 0.81,
                "timeliness_score": 1.0,
                "error_penalty": 0.0,
                "risk_flags": 0,
                "influence": 0.22,
            },
            {
                "agent": "Risk Validator",
                "phase": "Backtest_Execution",
                "quality_score": 0.86,
                "decision_confidence": 0.83,
                "timeliness_score": 1.0,
                "error_penalty": 0.0,
                "risk_flags": 0,
                "influence": 0.14,
            },
        ]

    def _sanitize_json_value(self, value: Any) -> Any:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, list):
            return [self._sanitize_json_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._sanitize_json_value(item) for key, item in value.items()}
        return value

    def _build_price_csv(self) -> str:
        rows = [
            "Date,Close,Adj Close",
            "2024-01-01,100,100",
            "2024-01-02,101,101",
            "2024-01-03,103,103",
            "2024-01-04,104,104",
            "2024-01-05,106,106",
            "2024-01-06,108,108",
            "2024-01-07,109,109",
            "2024-01-08,111,111",
            "2024-01-09,112,112",
            "2024-01-10,115,115",
        ]
        return "\n".join(rows) + "\n"


# Simple in-memory vector store (placeholder)
class InMemoryVectorStore:
    def __init__(self):
        self._items = []

    def add(self, key: str, vector: List[float], metadata: Optional[dict] = None):
        self._items.append((key, vector, metadata))

    def search(self, vector: List[float], k: int = 5):
        # naive: return up to k items
        return self._items[:k]


def get_vector_store(simulate: bool = True):
    if simulate:
        return InMemoryVectorStore()
    try:
        import faiss
        # user will configure a proper faiss index elsewhere
        return None
    except Exception:
        return InMemoryVectorStore()
