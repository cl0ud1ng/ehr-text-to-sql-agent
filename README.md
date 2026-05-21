# Lightweight EHR Text-to-SQL Agent

This project implements a lightweight medical Text-to-SQL agent for the local EHRSQL SQLite databases.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env` before running model-backed SQL generation.

## CLI

Run a raw SQL smoke test without an API key:

```bash
python -m src.cli --db mimic_iii --sql "select distinct route from PRESCRIPTIONS where lower(drug) = 'fluconazole'" --json
```

Run an agent query:

```bash
python -m src.cli --db mimic_iii --question "What is the method of fluconazole intake?"
```

Show retrieved schema:

```bash
python -m src.cli --db mimic_iii --question "What is the method of fluconazole intake?" --schema
```

## Evaluation

```bash
python -m src.evaluator --db mimic_iii --data "实验三材料/EHRSQL/测试集/mimic_iii_test_empty.json" --limit 10
```

## Web UI

After the core CLI works:

```bash
streamlit run src/app.py
```

The page supports database/model/prompt selection, independent query mode, follow-up mode with session summaries, test-sample loading, schema inspection, SQL/result display, repair logs, and single-sample execution comparison.
