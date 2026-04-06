# QuantLab-TDLC Refactor TODO

## Prompt 1 Baseline TODO

- Map the legacy ChatDev company roles to trading-domain roles.
- Replace the software-delivery chat chain with a trading lifecycle chain.
- Add a browsing-capable market research role.
- Replace test/fix recursion with backtest/optimization recursion driven by Sharpe and max drawdown thresholds.
- Add finance-aware tools for market research, paper discovery, price retrieval, and backtest execution.
- Create owner-facing metrics so agent contribution can be reviewed after each backtest.

## Prompt 2 Improvements

- Use `QuantLabConfig/` as the runtime-facing config home for prompts, schemas, and phase definitions.
- Shift the core paradigm from SDLC to a hypothesis-driven quantitative research loop.
- Rename the phases to:
  - `Alpha_Discovery`
  - `Model_Architecture`
  - `Signal_Engineering`
  - `Backtest_Execution`
  - `Recursive_Optimization`
- Add two explicit roles missing from classic ChatDev:
  - `Alpha Researcher`
  - `Risk Validator`
- Require strict JSON outputs from every agent so downstream routing and KPI scoring remain machine-readable.
- Optimize the role roster for local Ollama execution with sequential model loading on 12 GB VRAM.

## Repo-Specific Implementation Plan

- Keep the user-authored [blueprint_QuantLab.json](d:/kp_ai_agent/QuantLab_TDLC/blueprint_QuantLab.json) as the high-level design source.
- Add compatibility artifacts under `CompanyConfig/QuantTeam/` for the legacy prompt-1 terminology.
- Add runtime artifacts under `QuantLabConfig/` for the current YAML graph engine.
- Implement finance tools in `functions/function_calling/` instead of creating a legacy `BrowsingTool` class, because this repo exposes tools via public Python functions.
- Create a new workflow at `yaml_instance/QuantLab_TDLC.yaml` instead of modifying non-existent `RoleConfig.json` and `ChatChainConfig.json` runtime loaders.
- Keep the optimization loop bounded with a `loop_counter` guard.
- Make the `Risk Validator` responsible for both backtest evaluation and owner-facing agent KPI scoring.

## Deliverables

- `CompanyConfig/QuantTeam/RoleConfig.json`
- `CompanyConfig/QuantTeam/ChatChainConfig.json`
- `QuantLabConfig/agent_roster.json`
- `QuantLabConfig/phase_definitions.json`
- `QuantLabConfig/output_schemas.json`
- `yaml_instance/QuantLab_TDLC.yaml`
- `functions/function_calling/quant_trade.py`
- `tests/test_quant_trade_tools.py`

## Acceptance Checks

- Workflow YAML validates with the typed loader.
- Backtest tool returns Sharpe, Sortino, max drawdown, net profit, and pass/fail status.
- KPI tool returns agent-level contribution bands: `positive`, `low`, or `negative`.
- KPI tool also returns an owner-facing action recommendation: `retain`, `suspend`, or `dismiss`.