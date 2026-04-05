# QuantLab (ChatDev fork) End-to-End Recovery Plan

## 1. Project Overview

### What this system is supposed to do
- Core: zero-code workflow orchestration for multi-agent tasks via YAML graphs.
- Target workflow in this repo: QuantLab-TDLC (quant trading lifecycle).
- Agents in YAML are orchestrated as nodes in Directed Acyclic Graph / cycle with conditions.
- Includes research, strategy design, implementation, backtesting, optimization, and risk validation.
- Execution uses OpenAI-type LLM providers via `runtime` + function tools.

### High-level architecture
- Entry points:
  - `run.py` (CLI: load YAML, ask user prompt, execute GraphExecutor)
  - `server_main.py` (FastAPI backend + websocket orchestration)
- Core modules:
  - `workflow/` (GraphExecutor, GraphManager, CycleManager, runtime strategies)
  - `runtime/` (node executors, agent utilities, function calling engine, memory/thinking)
  - `entity/` (config schema objects, graph config, messages, edge rules)
  - `functions/function_calling` (tools exposed to agent nodes, including `quant_trade.py`)
  - `yaml_instance/` (workflow templates; `QuantLab_TDLC.yaml` is relevant)
- Data flow:
  - Input prompt -> graph start nodes -> node executors -> edges -> next nodes, loop via condition engine -> output at FINAL node.
- Agent interaction:
  - YAML nodes with type `agent` run LLM provider tasks, return JSON output.
  - Synchronous orchestration in GraphExecutor with condition checks and loops (`Optimization Guard`).
- External integrations:
  - LLM: OpenAI/Gemini/LM Studio/Ollama (`BASE_URL`, `API_KEY` from .env).
  - Databases: local filesystem Warehouse output; no DB required by default.
  - APIs: optional web search, Yahoo Finance, Arxiv etc (in quant_trade tools).

---

## 2. Current State Assessment

### What works
- Core runtime is valid Python and unit tests pass (`pytest 8 passed`).
- Graph engine, node executor classes, condition/processor framework present and strong.
- Quant trade functions (`run_strategy_backtest`, `evaluate_agent_kpis`) are implemented and robust.
- Server routes for workflow CRUD and running exist and are connected.

### What is broken (or not ready)
- YAML validation fails in local environment for all files due placeholder variables in `yaml_instance` (`${BASE_URL}`, `${API_KEY}` etc). This is expected until environment injection is done.
- No `make` command available in environment, but Python CLI scripts exist and run.
- `QuantLab_TDLC.yaml` likely requires realistic external LLM + network calls to complete; cannot run in CI without config.
- At runtime, if missing `BASE_URL` / `API_KEY`, config fails.
- No explicit mock provider path for offline local end-to-end tests unless you set a dummy provider or local model.
- Potential external dependency warnings: `requests` version mismatch in virtualenv.

---

## 3. Critical Issues (BLOCKERS)

1. Missing `.env` values: `BASE_URL`, `API_KEY` needed by graphs.
2. `QuantLab_TDLC.yaml` has LLM target models not guaranteed available; if provider fails, whole specification fails.
3. Incomplete w/o data source for specialty backtest: `Price history` is only pulled via symbol or CSV; some algorithms could fail on no data.
4. No task prompt provided when run with `run.py` in non-interactive mode (need automated injection or set arguments).
5. Node type compatibility may break on custom agent model names if provider unsupported in runtime.

---

## 4. Step-by-Step Recovery Plan

### Phase 1: Environment Setup

1. Ensure Python environment:
   - Python 3.12 (`>=3.12,<3.13`) per `pyproject.toml`.
   - Activate venv:
     ```powershell
     cd d:\kp_ai_agent\QuantLab_TDLC
     .venv-1\Scripts\Activate.ps1
     ```
2. Install dependency stack:
   - `pip install -r requirements.txt` (or `uv sync` if uv installed)
   - For frontend: `cd frontend && npm install`
3. Add `.env` from `.env.example`, set at least:
   ```env
   BASE_URL=https://api.openai.com/v1  # or local Ollama URL
   API_KEY=sk-xxx                        # or ollama
   ```
4. Ensure services are reachable.
   - If using local Ollama: `BASE_URL=http://localhost:11434/v1`, `API_KEY=ollama`.
5. Handle optional service keys (web search etc) only if needed.

### Phase 2: Code Fixes

1. Placeholder resolution: in `yaml_instance/QuantLab_TDLC.yaml` confirm `vars` reads from environment.
   - No code change needed if dotenv loaded; if issue persists, add `BASE_URL`/`API_KEY` to OS env too.

2. Strengthen robust defaults in `check/check.py` for var interpolation errors:
   - Add fallback path or clearer error message for missing vars.

3. Add minimal `dev` mock provider (recommended quick win):
   - Create `utils/mock_llm_provider.py` or extend runtime to support dummy agent text output.
   - This enables end-to-end path without paying API calls for local smoke tests.

4. In `functions/function_calling/quant_trade.py`, ensure `run_strategy_backtest` handles 0 data gracefully (already there).

5. For the `Risk Validator` loop, verify `Optimization Guard` is connected correctly in `yaml_instance/QuantLab_TDLC.yaml`.
   - Already exists, though set to 5 loops; confirm condition flow triggers.

6. (Optional) Adjust worker timeouts to avoid long hangs: in `server/services/session_execution.py` or executor thread management.

### Phase 3: Agent Orchestration Fix

1. Verify `GraphExecutor` lifecycle is completed:
   - `_build_memories_and_thinking()` sets each manager; no issues.
   - `run()` chooses `DagExecutionStrategy` unless cycles present.

2. Confirm cycle behavior for recursion path:
   - `Risk Validator` decision edges may route to Quant Architect / Algo Developer and then eventually back through `Risk Validator`.
   - `Optimization Guard` ensures no infinite loop.

3. Test manual execution of QuantLab flow in local engine:
   - Create minimal `task_prompt` and run via CLI (see commands below).

### Phase 4: Trading Pipeline Activation

1. Start from `Alpha Discovery Prompt` (literal node) + input instructions.
2. `Alpha Researcher` collects hypotheses (tools, LLM). Output JSON.
3. `Portfolio Manager` selects hypothesis and constraints.
4. `Quant Architect` builds the model architecture and signals.
5. `Algo Developer` writes strategy code & build files using file tools.
6. `Risk Validator` runs `run_strategy_backtest` + `evaluate_agent_kpis`.
7. Decision path:
   - `APPROVED` -> `FINAL` and completes.
   - `REVISE_MODEL` -> back to `Quant Architect` plus `Optimization Guard` counting.
   - `REVISE_IMPLEMENTATION` -> back to `Algo Developer` plus `Optimization Guard`.
8. `Optimization Guard` halts if 5 loops occur.

### Phase 5: Dashboard / Reporting

1. Final output from workflow via `GraphContext.final_message()` and archive in `WareHouse`.
2. Server event pipeline pushes data via websocket in `WorkflowRunService`.
3. Frontend reads `/api/workflows`, `/api/workflow/execute`, and status streams in `server/routes/websocket.py`.
4. Ensure result includes:
   - `decision`, `backtest_metrics`, `agent_kpis`, and `owner_summary`.
5. Instrument `logs/server.log` and `data/` path from `ResultArchiver` for manual review.

---

## 5. Execution Command

### CLI (quick local run)
```powershell
cd d:\kp_ai_agent\QuantLab_TDLC
.venv-1\Scripts\Activate.ps1
python run.py --path yaml_instance/QuantLab_TDLC.yaml --name quantlab
```
- Enter the prompt when asked, e.g.: `"Run a quick proof-of-concept trade strategy in the dataset."`

### Server + Frontend (interactive)
```powershell
cd d:\kp_ai_agent\QuantLab_TDLC
.venv-1\Scripts\Activate.ps1
python server_main.py --port 6400 --reload
```
Frontend:
```powershell
cd frontend
VITE_API_BASE_URL=http://localhost:6400 npm run dev
```

### Direct API execute (fast path)
1. Create a session through websocket API (see `server/routes/sessions.py`).
2. POST `/api/workflow/execute` with payload:
```json
{
  "session_id": "s1",
  "yaml_file": "QuantLab_TDLC.yaml",
  "task_prompt": "Run a baseline quant strategy using SPY and produce JSON output.",
  "variables": {
    "BASE_URL": "https://api.openai.com/v1",
    "API_KEY": "sk-..."
  }
}
```

---

## 6. Definition of DONE

1. `python run.py --path yaml_instance/QuantLab_TDLC.yaml` completes with no uncaught traceback.
2. `GraphExecutor` returns all node outputs and the final node `FINAL` string has valid JSON.
3. Backtest step from `Risk Validator` returns a forecast throughput JSON with at least metrics `sharpe_ratio`, `max_drawdown`, `total_return`.
4. `evaluate_agent_kpis` returns non-empty `agent_kpis` and `owner_summary`.
5. Decision in final output is one of {`APPROVED`,`REVISE_MODEL`,`REVISE_IMPLEMENTATION`}.
6. If using server path, front-end can start workflow and display a successful status message.

---

## 7. Assumptions and Next Actions

- This plan assumes an operational LLM endpoint with correct keys. If not available, run with a local mock provider for agents.
- For local validation, ensure `BASE_URL` and `API_KEY` are set as env vars to resolve workflow placeholders.
- The first actionable recovery step is to run the CLI with those env vars and observe logs; then tune agent prompt JSON compliance and tool paths.
