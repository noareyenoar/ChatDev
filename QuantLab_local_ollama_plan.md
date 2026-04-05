# QuantLab Local Ollama Execution Plan

## Goal

Start QuantLab in a way that is executable on this machine now, using the local Ollama server as the primary LLM backend, while keeping an offline fallback for workflow and tool-chain validation.

This plan merges the stronger parts of the existing recovery notes:
- keep the practical repo-state assessment from the FreeAgent plan
- keep the simpler staged rollout from the original recovery plans
- narrow scope to local Ollama first, not live trading or broker execution

## Verified Local Baseline

The current workspace already contains several enabling changes:
- `yaml_instance/QuantLab_TDLC.yaml` resolves provider vars from `BASE_URL` and `API_KEY`
- `check/check.py` already supports CLI validation with `--vars` and `--var`
- Launch API and UI already support per-run variable overrides
- `yaml_instance/QuantLab_TDLC_simulated.yaml` and the simulated provider already exist for offline smoke tests

The local Ollama server is reachable at `http://127.0.0.1:11434` and currently exposes these relevant models:
- `llama3.1:8b`
- `mistral-nemo:12b`
- `qwen2.5:7b`
- `qwen2.5-coder:7b`
- `llama3-groq-tool-use:8b`
- `deepseek-r1:8b`
- `deepseek-r1:14b`

## Chosen Execution Strategy

Use a dedicated local workflow variant instead of forcing the generic QuantLab YAML to fit every environment.

Why this path is the right one:
- the original QuantLab YAML references one model tag that is not installed here: `qwen2.5-math:7b`
- the original Alpha Researcher model tag is more specific than the locally installed Ollama tag: `llama3.1:8b-instruct-q5_K_M` versus `llama3.1:8b`
- Ollama is more reliable through Chat Completions compatibility than through the Responses API, so the local workflow should force `params.protocol: chat`
- local-first operation should avoid optional web-search keys unless the user explicitly adds them later

## Scope For This First Working Pass

Included now:
- local Ollama CLI workflow path
- local Launch UI path using variable overrides
- YAML validation path
- offline simulated fallback path
- docs for startup, validation, and troubleshooting

Deferred until after a stable first run:
- live broker or paper-trading connectors
- metrics dashboard improvements
- stricter Risk Validator contract enforcement
- loop-routing regression tests
- Python 3.12 migration of the active environment

## Concrete Model Mapping

Map QuantLab roles to installed local models as follows:

| Role | Local model | Reason |
| --- | --- | --- |
| Alpha Researcher | `llama3.1:8b` | available, general-purpose reasoning, acceptable for research summarization |
| Portfolio Manager | `mistral-nemo:12b` | available and stronger general planner than the 7B tier |
| Quant Architect | `qwen3.5:7b` | available replacement for missing `qwen2.5-math:7b` |
| Algo Developer | `qwen2.5-coder:7b` | already installed and task-aligned |
| Risk Validator | `qwen2.5-coder:7b` | installed and good enough for tool-based evaluation routing |

## Execution Phases

### Phase 1. Stabilize The Local Ollama Workflow

1. Create `yaml_instance/QuantLab_TDLC_ollama_local.yaml`.
2. Point every agent to `provider: openai` with:
   - `base_url: ${OLLAMA_BASE_URL}`
   - `api_key: ${OLLAMA_API_KEY}`
   - `params.protocol: chat`
3. Replace unavailable or mismatched model tags with installed local tags.
4. Remove optional search tools that depend on extra API keys from the first-pass researcher node.

### Phase 2. Document The Operator Path

1. Write a guide that assumes:
   - Ollama host: `http://127.0.0.1:11434`
   - API-compatible base URL: `http://127.0.0.1:11434/v1`
   - API key placeholder: `ollama`
2. Document both:
   - direct CLI run
   - server and Launch UI run
3. Document the simulated fallback when the local model path is slow or unstable.

### Phase 3. Validate And Execute

1. Validate the new YAML with explicit vars overrides.
2. Hit the Ollama OpenAI-compatible chat endpoint with a small direct request.
3. Run the new local workflow from CLI with a short deterministic prompt.
4. If it fails, fix the root cause if it is repo-local and practical.

### Phase 4. Decide What Needs User Input

Only ask the user for input if one of these blocks execution:
- the target Ollama model should be changed for performance reasons
- the local Python 3.11 environment breaks runtime behavior that cannot be worked around safely
- optional web-search keys are desired for better research breadth

## First Definition Of Done

This first pass is done when all of the following are true:

1. The new local Ollama YAML validates successfully.
2. The local Ollama server responds on the OpenAI-compatible path.
3. A CLI execution starts against the local YAML.
4. The guide documents the exact commands and model mapping for this machine.
5. There is a clean fallback path using `yaml_instance/QuantLab_TDLC_simulated.yaml`.

## Known Risks

- The active venv is Python `3.11.9`, while the project declares `>=3.12,<3.13`.
- Local model latency may be high, especially for multi-node tool-enabled runs.
- Quant research tools still depend on outbound internet access for Yahoo Finance, ArXiv, and DDGS.
- The first real local-LM run may still need prompt or tool-scope tightening if a node overuses external tools.

## Immediate Next Actions

1. Add the local Ollama workflow variant.
2. Add the running guide.
3. Validate the new YAML.
4. Execute the CLI run against local Ollama.