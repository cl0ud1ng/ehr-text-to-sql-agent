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

Use deterministic representative examples for few-shot prompting:

```bash
python -m src.cli --db mimic_iii --prompt-version fewshot --example-type time --time-tags exact-first --question "When was the first prescription?"
```

Show retrieved schema:

```bash
python -m src.cli --db mimic_iii --question "What is the method of fluconazole intake?" --schema
```

## Evaluation

```bash
python -m src.evaluator --db mimic_iii --data "data/EHRSQL/测试集/mimic_iii_test_empty.json" --limit 10
```

Run samples in parallel and write JSON/CSV artifacts:

```bash
python -m src.evaluator \
  --db mimic_iii \
  --data outputs/evaluation/small50_mimic_iii_samples.json \
  --workers 6 \
  --output outputs/evaluation/small50_mimic_iii_results.json \
  --rows-csv outputs/evaluation/small50_mimic_iii_rows.csv \
  --summary-output outputs/evaluation/small50_mimic_iii_summary.json \
  --summary-csv outputs/evaluation/small50_mimic_iii_summary.csv
```

Summarize existing result JSON files without re-running the model:

```bash
python -m src.evaluator \
  --summarize outputs/evaluation/small50_mimic_iii_results_fixed_parallel.json outputs/evaluation/small50_eicu_results_fixed_parallel.json \
  --summary-output outputs/evaluation/small50_grouped_summary.json \
  --summary-csv outputs/evaluation/small50_grouped_summary.csv
```

## Repair Demo

Run deterministic SQL repair cases and save the report artifact:

```bash
python -m src.repair_demo --json
```

The summary is written to `outputs/evaluation/repair_cases.json`; detailed run logs are written under `outputs/runs/`.

## Web UI

After the core CLI works:

```bash
streamlit run src/app.py
```

The page supports database/model/prompt selection, independent query mode, follow-up mode with session summaries, test-sample loading, schema inspection, SQL/result display, repair logs, and single-sample execution comparison.
