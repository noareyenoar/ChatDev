# QuantLab Running Guide

## Purpose

This guide is the shortest reliable path to run QuantLab on this machine using the local Ollama server first, with an offline simulated fallback when needed.

## Local Environment Assumptions

Verified from the current machine:
- Ollama host: `http://0.0.0.0:11434`
- Reachable local endpoint: `http://127.0.0.1:11434`
- OpenAI-compatible base URL for this project: `http://127.0.0.1:11434/v1`
- Recommended placeholder API key: `ollama`
- Active Python environment: `.venv`
- Active Python version in that environment: `3.11.9`

Important note:
- the project declares Python `>=3.12,<3.13`
- the current venv is still usable for validation and may run the workflow, but it is not the declared target runtime

## Models Available In Local Ollama

These model tags are already installed and usable now:
- `llama3.1:8b`
- `mistral-nemo:12b`
- `qwen3.5:9b`
- `qwen2.5-coder:7b`
- `llama3-groq-tool-use:8b`
- `phi4-mini-reasoning:3.8b`
- `deepseek-r1:8b`
- `deepseek-r1:14b`

The local workflow variant currently runs in hybrid stability mode:
- Alpha Researcher: `llama3-groq-tool-use:8b`
- Portfolio Manager: `qwen3.5:9b`
- Quant Architect: `qwen3.5:9b`
- Algo Developer: `qwen2.5:7b`
- Risk Validator: `llama3-groq-tool-use:8b`

Reason:
- keeps a tool-capable model for the research stage where internet mining may be needed
- lowers CPU and RAM spikes
- avoids model-specific tool support mismatches
- always emits a final decision JSON even when validator requests revisions

Notes:
- non-coder qwen roles use `qwen3.5:9b`
- strategy contract is pinned to `moving_average_crossover` for backtest compatibility

## Optional GPU Runtime Profile

If you start Ollama with your GPU profile script, this guide remains compatible. The profile in [../zenv_list_to_use_GPU](../zenv_list_to_use_GPU) sets DirectML and Vulkan-related env vars and then starts `ollama serve`.

Recommended for your machine stability:
- Keep `OLLAMA_NUM_GPU=1` and sequential workflow execution.
- Prefer 7B-8B models for tool-heavy nodes.
- Keep context windows limited (the workflow now uses reduced context windows).
- Keep token budgets bounded to prevent runaway CPU/RAM usage.

## Recommended Environment Variables

Put these values in `.env` if you want repo-wide defaults:

```env
BASE_URL=http://127.0.0.1:11434/v1
API_KEY=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_API_KEY=ollama
```

Why both pairs are shown:
- the repo generally resolves model settings from `BASE_URL` and `API_KEY`
- the dedicated local workflow also exposes `OLLAMA_BASE_URL` and `OLLAMA_API_KEY` vars for clarity

## Files To Use

Use these workflows depending on what you want to verify:
- `yaml_instance/QuantLab_TDLC_ollama_local.yaml`: real local Ollama execution path
- `yaml_instance/QuantLab_TDLC_simulated.yaml`: offline deterministic smoke test with no external LLM
- `yaml_instance/QuantLab_TDLC.yaml`: original generic QuantLab workflow

## Local Dataset Folders

This workflow now prioritizes datasets in:
- `raw_data_and_backtest/raw_data`
- `raw_data_and_backtest/prepared_dataset`
- `raw_data_and_backtest/data`

Known starter files that already exist:
- `raw_data_and_backtest/prepared_dataset/BTCUSDT.csv`
- `raw_data_and_backtest/data/BTCUSDT.csv`

Backtest-compatible file prepared during setup:
- `raw_data_and_backtest/prepared_dataset/BTCUSDT_backtest.csv`
  - columns: `Date`, `AdjClose`

Agents are also configured to browse internet sources for additional dataset mining when needed.

## 1. Validate The Local Ollama Workflow

Run from the repo root:

```powershell
d:/kp_ai_agent/QuantLab_TDLC/.venv/Scripts/python.exe -m check.check --path yaml_instance/QuantLab_TDLC_ollama_local.yaml --var OLLAMA_BASE_URL=http://127.0.0.1:11434/v1 --var OLLAMA_API_KEY=ollama
```

If this passes, the YAML and placeholder resolution are structurally valid.

## 2. Quick Ollama Endpoint Check

Use PowerShell to confirm the OpenAI-compatible chat path responds:

```powershell
$body = @{
  model = 'llama3.1:8b'
  messages = @(
    @{ role = 'user'; content = 'Reply with the single word READY.' }
  )
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Uri http://127.0.0.1:11434/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
```

Expected outcome:
- HTTP success
- one assistant message containing `READY` or similar

## 3. Run QuantLab From CLI With Local Ollama

Use a short prompt for the first run:

```powershell
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434/v1'
$env:OLLAMA_API_KEY = 'ollama'

@'
Design a simple SPY trend-following strategy, backtest it, and return strict JSON.
'@ | d:/kp_ai_agent/QuantLab_TDLC/.venv/Scripts/python.exe run.py --path yaml_instance/QuantLab_TDLC_ollama_local.yaml --name quantlab_ollama_local
```

What this does:
- loads the local workflow variant
- uses the local Ollama server via the OpenAI-compatible API
- asks each node to keep output machine-readable

## 4. Run The Offline Fallback

If the local Ollama run is too slow or unstable, verify the orchestration path first:

```powershell
@'
offline quantlab simulation
'@ | d:/kp_ai_agent/QuantLab_TDLC/.venv/Scripts/python.exe run.py --path yaml_instance/QuantLab_TDLC_simulated.yaml --name quantlab_sim
```

This does not use Ollama.

## 5. Run Backend And Frontend

Backend:

```powershell
d:/kp_ai_agent/QuantLab_TDLC/.venv/Scripts/python.exe server_main.py --port 6400 --reload
```

Frontend:

```powershell
cd frontend
$env:VITE_API_BASE_URL = 'http://localhost:6400'
npm run dev
```

Then open the Launch page and choose `QuantLab_TDLC_ollama_local.yaml`.

If you do not want to edit `.env`, use the Launch settings dialog and set workflow variable overrides to:

```json
{
  "OLLAMA_BASE_URL": "http://127.0.0.1:11434/v1",
  "OLLAMA_API_KEY": "ollama"
}
```

## 6. Troubleshooting

### Validation fails on missing vars

Use `--var` overrides during validation or add the variables to `.env`.

### Workflow loads but model call fails

Check that the requested model tag exists in Ollama:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:11434/api/tags | ConvertTo-Json -Depth 8
```

### Ollama is up but responses are slow

Use one of these options:
- switch the first pass to `yaml_instance/QuantLab_TDLC_simulated.yaml`
- reduce prompt complexity
- change one or more roles to a lighter installed model such as `phi4-mini-reasoning:3.8b` or `qwen2.5:7b`

### Research tools fail

That usually means network reachability or optional search APIs are unavailable. The local workflow reduces optional tool usage, but Yahoo Finance, ArXiv, and DDGS still require outbound internet access.

### Python version mismatch

If runtime errors appear that are hard to explain, create a Python 3.12 environment and reinstall dependencies before debugging deeper.

## What Was Prepared For This Machine

This repo now has a practical local-Ollama run path centered on:
- `yaml_instance/QuantLab_TDLC_ollama_local.yaml`
- this guide
- the simulated fallback workflow

If you want, the next step after the first successful run is to tune the role-to-model mapping for speed versus output quality.