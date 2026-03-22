## Plan: Run QuantLab TDLC End-to-End

TL;DR - เป้าหมาย: รัน yaml_instance/QuantLab_TDLC.yaml กับโมเดลที่ถูก serve โดย Ollama โดยทดสอบแบบโลคัล (Windows) ให้ตรวจสอบ env, deps, validator, tests และรัน workflow แบบ non-interactive เพื่อยืนยันว่า agent chain ทำงานครบวงจร

**Steps**
1. **Prepare Python**: ติดตั้ง Python 3.12 (หากยังไม่มี). สร้าง virtualenv และ activate.
   - PowerShell: `python -m venv .venv` ; `./.venv/Scripts/Activate.ps1`
2. **Install deps**: `pip install --upgrade pip` ; `pip install -r requirements.txt`.
3. **Set Ollama env**: กำหนดตัวแปร `OLLAMA_BASE_URL` และถ้ามี `OLLAMA_API_KEY` (PowerShell):
   - `$Env:OLLAMA_BASE_URL='http://localhost:11434/v1'`
   - `$Env:OLLAMA_API_KEY='your_key_or_empty'`
   - (ถ้าต้องการรันแบบชั่วคราว ให้ตั้ง `TASK_PROMPT` เพื่อส่ง prompt เริ่มต้น)
4. **Run structural checks**: `python tools/validate_all_yamls.py` — แก้ YAML ถ้าพบข้อผิดพลาด.
5. **Run unit tests**: `pytest -v` — แก้ปัญหาข้อผิดพลาดก่อนรันจริง.
6. **Start backend (optional)**: ถ้าต้องการ API mode: `python server_main.py --port 6400 --reload` หรือ `uvicorn server.app:app --host 0.0.0.0 --port 6400 --reload`.
7. **Run TDLC workflow (non-interactive)**: กำหนด env แล้วรัน runner:
   - PowerShell example:
     - `$Env:OLLAMA_BASE_URL='http://localhost:11434/v1' ; $Env:OLLAMA_API_KEY='...' ; $Env:TASK_PROMPT='Run full QuantLab TDLC test' ; python run.py --path yaml_instance/QuantLab_TDLC.yaml --name QuantLabTest`
   - CMD example: `set OLLAMA_BASE_URL=http://localhost:11434/v1 && set OLLAMA_API_KEY=... && set TASK_PROMPT=Run full QuantLab TDLC test && python run.py --path yaml_instance/QuantLab_TDLC.yaml --name QuantLabTest`
8. **Observe outputs**: ตรวจสอบ stdout/logs, ผลลัพธ์ใน QuantLabConfig/output files, และ reports ที่บันทึกไว้โดยเครื่องมือ (paths ในรายงาน backtest).
9. **Verify agent KPIs**: รัน `tests/test_quant_trade_tools.py` และตรวจสอบ owner_summary ในรายงาน backtest ของ `quant_trade.evaluate_agent_kpis`.
10. **Troubleshooting & repeat**: หากพบปัญหาเกี่ยวกับ packages (เช่น faiss, cartopy) ให้ใช้ Docker flow: `docker compose -f compose.yml up --build`.

**Relevant files**
- run.py
- server_main.py
- server/app.py
- yaml_instance/QuantLab_TDLC.yaml
- QuantLabConfig/agent_roster.json
- QuantLabConfig/phase_definitions.json
- blueprint_QuantLab.json
- requirements.txt
- pyproject.toml
- .env.example
- tools/validate_all_yamls.py
- functions/function_calling/quant_trade.py
- tests/test_quant_trade_tools.py
- compose.yml
- Dockerfile

**Verification**
1. `python tools/validate_all_yamls.py` → no errors
2. `pytest -v` → all tests pass
3. Run `python run.py --path yaml_instance/QuantLab_TDLC.yaml --name QuantLabTest` with Ollama env → workflow completes without uncaught exceptions and produces backtest/KPI reports
4. Confirm reports serialized (no Path/serialization errors) and that `evaluate_agent_kpis` yields sensible `owner_summary` and `recommended_action` fields

**Decisions / Assumptions**
- ใช้ Python 3.12 ตาม `pyproject.toml`.
- Ollama ให้บริการที่ `http://localhost:11434/v1` หรือ URL ที่ผู้ใช้กำหนด
- หากการติดตั้งบาง dependency ติดปัญหาบน Windows ให้รันผ่าน Docker

**Further Considerations**
1. If you want, I can generate a PowerShell script that automates venv creation, env var setting, install, validation, and a single-run of the YAML. (Recommended for repeatable local tests.)
2. For production-like runs, prepare a data source (CSV or DB) and ensure `QuantLabConfig/agent_roster.json` hardware_profile matches your Ollama model memory/VRAM.
3. If you want, I can attempt to run a smoke test here (create venv and attempt to run `python run.py`), but I need permission to execute commands in this environment.

-- End of plan
