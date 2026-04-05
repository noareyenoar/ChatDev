# plan_4_run.md

## 1. Project Overview

### What this system is supposed to do
QuantLab in this repository is a workflow-driven multi-agent runtime (forked from ChatDev and DevAll) where YAML-defined agent graphs orchestrate:
- Research and hypothesis generation
- Strategy design
- Implementation support
- Backtesting and KPI scoring
- Final governance decision loops (approve or revise)

Current trading implementation is simulation and backtest oriented, not live broker execution.

### High-level architecture
- Runtime entry points:
  - run.py: CLI graph runner
  - server_main.py: FastAPI server
  - runtime/sdk.py: Python SDK entry
- Workflow definition:
  - yaml_instance/QuantLab_TDLC.yaml
- Graph execution engine:
  - workflow/graph.py
  - workflow/graph_manager.py
  - workflow/runtime/*
- Config validation and loading:
  - check/check.py
  - check/check_workflow.py
- Tooling layer:
  - functions/function_calling/quant_trade.py
  - utils/function_manager.py
- Server orchestration:
  - server/routes/execute.py
  - server/services/workflow_run_service.py
  - server/services/websocket_manager.py
- Frontend surfaces:
  - frontend/src/pages/LaunchView.vue
  - frontend/src/pages/BatchRunView.vue

### Data and control flow (actual)
1. User selects YAML and prompt (Launch page or SDK and CLI).
2. YAML is loaded via check/check.py and placeholders are resolved via utils/vars_resolver.py.
3. Graph is built and executed by workflow/graph.py.
4. Agent nodes call function tools from functions/function_calling.
5. Risk Validator emits decision JSON; edges route to FINAL or revision nodes in yaml_instance/QuantLab_TDLC.yaml.
6. Outputs, logs, and artifacts are written to WareHouse session folders.

## 2. Current State Assessment

### What works now
- Core graph runtime executes and supports DAG and cycle patterns.
- Quant tool functions are present and wired:
  - search_market_intelligence
  - search_arxiv_papers
  - fetch_price_history
  - run_strategy_backtest
  - evaluate_agent_kpis
- Tests currently pass in this environment:
  - 8 passed
- Quant workflow can load when variables are explicitly overridden (verified).
- FastAPI entry import works (server_main import probe succeeded).

### What is broken or risky now
- CLI Quant workflow launch fails by default due unresolved and self-referential placeholders in yaml_instance/QuantLab_TDLC.yaml when .env is absent.
- Launch UI path cannot pass variable overrides to execute endpoint, so Quant workflow cannot start from Launch unless environment is preconfigured.
- YAML bulk validator gives false confidence because it shells to python -m check.check, but check/check.py has no CLI main entrypoint.
- Local environment Python version is 3.11.9 while project requires >=3.12,<3.13.
- Dependency warning indicates requests stack version incompatibility (urllib3 and chardet warning).
- QuantLabConfig JSON assets are not referenced by backend runtime (dead configuration artifacts).
- No live execution connector (broker and exchange order placement) is implemented; only research and backtest loop is operational.

## 3. Critical Issues (BLOCKERS)

1. Quant workflow variable cycle and missing env setup
- File: yaml_instance/QuantLab_TDLC.yaml
- Symptom: ConfigError detected placeholder cycle referencing OLLAMA_BASE_URL when environment keys are absent.
- Impact: run.py cannot execute Quant workflow out of the box.

2. Missing environment bootstrap for Quant mode
- Files: .env.example, README.md
- Symptom: OLLAMA_BASE_URL and OLLAMA_API_KEY are required by Quant YAML but not documented in default quick-start path.
- Impact: users cannot reliably start Quant pipeline.

3. Validator pipeline is misleading
- Files: tools/validate_all_yamls.py, check/check.py
- Symptom: validate_all_yamls invokes python -m check.check --path ... but check.check has no executable CLI main.
- Impact: reported YAML pass status may not represent real validation execution.

4. Launch mode missing vars override channel
- Files: frontend/src/pages/LaunchView.vue, server/routes/execute.py, server/models.py
- Symptom: Launch payload excludes variables field; backend execute route does not accept variables.
- Impact: Quant workflows requiring runtime vars fail unless process env is pre-populated.

5. Environment compatibility mismatch
- File: pyproject.toml
- Symptom: requires Python 3.12+, current environment is 3.11.9.
- Impact: latent runtime and package breakages and non-reproducible behavior.

6. Not truly end-to-end trading execution
- File: functions/function_calling/quant_trade.py
- Symptom: no broker and exchange order-routing tools or execution sink.
- Impact: system can simulate and evaluate strategy but cannot execute live or paper trades via exchange API.

## 4. Step-by-Step Recovery Plan

### Phase 1: Environment Setup

Goal: make startup deterministic and reproducible.

1. Python and runtime alignment
- Create or switch to Python 3.12 environment.
- Reinstall backend dependencies with uv sync.

2. Required environment keys
- Create .env with at least:
  - BASE_URL
  - API_KEY
  - OLLAMA_BASE_URL
  - OLLAMA_API_KEY
- For local Ollama compatibility, set OLLAMA_BASE_URL to local OpenAI-compatible endpoint and OLLAMA_API_KEY to a non-empty placeholder.

3. Dependency hygiene
- Pin compatible requests stack (requests, urllib3, chardet trio) to remove warning and reduce unpredictable HTTP behavior.

4. External services readiness
- Ensure chosen model provider endpoint is reachable before running workflow.
- Ensure internet access for market and research tools (Yahoo, ArXiv, Serper and Jina if used).

### Phase 2: Code Fixes

Goal: remove hard blockers with minimal edits.

1. Fix Quant YAML variable fallback
- Edit yaml_instance/QuantLab_TDLC.yaml:
  - Replace self-referential vars with default provider vars (for example BASE_URL and API_KEY), or provide safe literal defaults.
- Expected result: load_config succeeds without placeholder-cycle failure.

2. Add executable CLI entry to checker
- Edit check/check.py:
  - Add argparse-based main accepting --path, --fn-module, and optional --vars JSON.
  - Execute load_config and print explicit pass and fail.
  - Return non-zero on failure.
- Expected result: validation tooling reflects real status.

3. Repair bulk validator behavior
- Edit tools/validate_all_yamls.py:
  - Keep subprocess call only after check.check CLI is real.
  - Capture stderr and include failing reason per file.
- Expected result: trustworthy YAML quality gate.

4. Enable Launch vars override path
- Edit server/models.py: add optional variables to WorkflowRequest.
- Edit server/routes/execute.py: pass variables into workflow run service.
- Edit server/services/workflow_run_service.py: pass vars_override to load_config.
- Edit frontend/src/pages/LaunchView.vue: add optional vars input and include in execute payload.
- Expected result: Quant workflow can run from UI without relying only on process-level env.

5. Document Quant run prerequisites
- Edit README.md and docs user guide:
  - Quant workflow env keys
  - local model assumptions
  - run commands for CLI and API and UI
- Expected result: reproducible operator path.

### Phase 3: Agent Orchestration Fix

Goal: ensure recursive loop works reliably and terminates safely.

1. Validate Risk Validator decision contract
- Enforce strict JSON output and decision enum check before edge routing.
- If parse fails, route to retry and fail-safe node instead of silent dead-end.

2. Verify loop guard behavior
- Confirm yaml_instance/QuantLab_TDLC.yaml branch logic from Risk Validator to Optimization Guard and FINAL.
- Ensure max_iterations path always reaches FINAL with explicit HALT payload.

3. Add regression tests for orchestration decisions
- Add tests covering APPROVED, REVISE_MODEL, REVISE_IMPLEMENTATION, and guard termination.

### Phase 4: Trading Pipeline Activation

Goal: get data to strategy to backtest to decision loop fully runnable.

1. Define minimal runnable success path (simulation mode)
- Input prompt
- Alpha Researcher research output
- Architect strategy spec
- Developer backtest contract
- Risk Validator executes run_strategy_backtest plus evaluate_agent_kpis
- FINAL returns APPROVED or HALT with machine-readable metrics

2. Stabilize tool-call reliability
- Add retries and timeouts around external HTTP calls in quant tools where missing.
- For unavailable network, support fallback path using user-supplied CSV in run_strategy_backtest.

3. Optional paper and live execution extension (separate track)
- Add broker adapter tools (paper first) and explicit risk guardrails.
- Keep disabled by default; retain simulation mode as baseline definition of done.

### Phase 5: Dashboard and Reporting

Goal: make output visible and auditable for operators.

1. Surface Quant metrics in Launch view
- Parse final JSON and render key metrics cards:
  - total_return
  - sharpe_ratio
  - sortino_ratio
  - max_drawdown
  - net_profit
  - decision

2. Persist and expose run artifacts
- Ensure risk report JSON is always saved in session output folder.
- Add convenient download links from existing artifact APIs.

3. Add execution summary panel
- Show per-agent KPI table and owner summary actions (retain and suspend and dismiss).

## 5. Execution Command

Minimal local run sequence after fixes:

~~~bash
# 1) Python 3.12 environment and dependencies
uv sync

# 2) Frontend dependencies
cd frontend
npm install
cd ..

# 3) Create env file
copy .env.example .env
# then set BASE_URL and API_KEY and OLLAMA_BASE_URL and OLLAMA_API_KEY

# 4) Validate workflows (after checker CLI fix)
uv run python tools/validate_all_yamls.py

# 5) Start backend
uv run python server_main.py --port 6400

# 6) Start frontend (new terminal)
cd frontend
VITE_API_BASE_URL=http://localhost:6400 npm run dev

# 7) Run Quant workflow from UI (Launch) or SDK
# UI: select QuantLab_TDLC.yaml and submit task prompt
~~~

CLI option for direct run once env is fixed:

~~~bash
uv run python run.py --path yaml_instance/QuantLab_TDLC.yaml --name quantlab_smoke
~~~

## 6. Definition of DONE

System is DONE when all criteria below pass:

1. Startup and validation
- Backend and frontend start without fatal errors.
- YAML validator fails on intentionally invalid YAML and passes on valid YAML.

2. Quant workflow execution
- QuantLab workflow starts from Launch UI and from CLI and SDK.
- Workflow completes to FINAL without manual code patching during run.

3. Observable outputs
- FINAL output is strict JSON including:
  - decision
  - backtest_metrics
  - agent_kpis
  - owner_summary
- Session artifacts are saved and downloadable.

4. Trading simulation objective met
- At least one simulated backtest run completes and returns metrics plus governance decision.

5. Stability checks
- Existing tests pass.
- Added regression tests for decision routing and loop guard pass.

## Assumptions

- Target first milestone is reliable simulation and backtest pipeline, not immediate live broker execution.
- Local or remote LLM endpoint is available and reachable.
- Quant workflow remains YAML-driven under current DevAll runtime rather than a separate dedicated trading microservice.
