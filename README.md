# Lightweight EHR Text-to-SQL Agent

本项目实现了一个面向本地 EHRSQL SQLite 数据库的轻量级医疗 Text-to-SQL Agent。系统从自然语言问题出发，完成 Schema 检索、可回答性判断、SQL 生成、只读校验与执行、错误修复、结果摘要、批量评估和 Streamlit 可视化展示。

## 功能概览

- 支持 `mimic_iii` 和 `eicu` 两个 EHRSQL SQLite 数据库。
- 自动读取真实数据库 Schema，并按问题检索候选表和字段。
- 支持规则与 LLM 结合的可回答性判断，对数据库外问题拒答。
- 支持 `base`、`schema`、`fewshot`、`reflection` 四类 prompt。
- 默认使用 deterministic few-shot 示例，示例来自 `data/EHRSQL/示例数据/`。
- 内置常见 EHRSQL 模板规则，可离线覆盖部分 route、cost、stay length 和时间推理样例。
- 对 SQL 做单条只读查询校验，拒绝写操作、多语句和未知表。
- 使用只读 SQLite 连接执行 SQL，并限制超时和最大返回行数。
- 校验或执行失败时最多自动修复 2 轮，并记录修复日志。
- 提供 CLI、批量 evaluator、自动修复 demo 和 Streamlit Web UI。
- 运行日志写入 `outputs/runs/`，模型缓存写入 `outputs/cache/`。

## 目录结构

```text
.
├── configs/
│   └── default.yaml              # 数据库路径、默认模型和 Agent 参数
├── data/EHRSQL/
│   ├── mimic_iii.sqlite          # 本地 MIMIC-III SQLite 数据库
│   ├── eicu.sqlite               # 本地 eICU SQLite 数据库
│   ├── 测试集/                    # EHRSQL 测试集
│   └── 示例数据/                  # few-shot 代表样例
├── prompts/                      # base/schema/fewshot/reflection prompt
├── src/
│   ├── agent/                    # Agent 编排、生成、可回答性和修复
│   ├── app.py                    # Streamlit Web UI
│   ├── cli.py                    # 单条查询与 raw SQL 入口
│   ├── evaluator.py              # 批量评估与指标汇总
│   ├── repair_demo.py            # 自动修复演示案例
│   ├── schema_index.py           # Schema 加载与检索
│   ├── sql_executor.py           # 只读 SQLite 执行
│   └── sql_guard.py              # SQL 静态安全校验
├── tests/                        # 单元测试
├── .env.example
├── requirements.txt
└── README.md
```

`outputs/`、`.env`、`.venv/`、`*.sqlite` 等本地产物已在 `.gitignore` 中忽略。新环境需要自行放置数据库文件，或确认当前工作区已有：

```text
data/EHRSQL/mimic_iii.sqlite
data/EHRSQL/eicu.sqlite
```

## 环境配置

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，至少设置：

```text
DEEPSEEK_API_KEY=your_deepseek_api_key
```

默认配置位于 `configs/default.yaml`。当前默认模型是 `deepseek-v4-flash`，也可以通过 `.env` 的 `DEEPSEEK_MODEL` 或命令行 `--model` 覆盖。raw SQL 执行和部分模板规则不需要 API key；未命中模板的 SQL 生成、LLM 可回答性判断和非模板修复需要 DeepSeek API。

## CLI 使用

执行一条 raw SQL，用于检查数据库连接和 SQL 执行器，不需要 API key：

```bash
python -m src.cli --db mimic_iii \
  --sql "select distinct route from PRESCRIPTIONS where lower(drug) = 'fluconazole'" \
  --json
```

运行单条 Agent 查询：

```bash
python -m src.cli --db mimic_iii \
  --question "What is the method of fluconazole intake?"
```

查看问题对应的候选 Schema：

```bash
python -m src.cli --db mimic_iii \
  --question "What is the method of fluconazole intake?" \
  --schema
```

使用时间推理 few-shot 示例：

```bash
python -m src.cli --db mimic_iii \
  --prompt-version fewshot \
  --example-type time \
  --time-tags exact-first \
  --question "When was the first prescription?"
```

常用 CLI 参数：

- `--db {mimic_iii,eicu}`：选择数据库。
- `--prompt-version {base,schema,fewshot,reflection}`：选择 prompt，默认 `fewshot`。
- `--example-type {auto,basic,time}`：few-shot 示例类型，默认自动判断。
- `--mode {new_query,followup}`：查询模式，CLI 默认 `new_query`。
- `--max-rows`、`--timeout`：控制 SQL 返回行数和超时。
- `--no-cache`：禁用模型缓存。
- `--json`：输出机器可读 JSON。

## 批量评估

在测试集上运行小批量评估：

```bash
python -m src.evaluator \
  --db mimic_iii \
  --data "data/EHRSQL/测试集/mimic_iii_test_empty.json" \
  --limit 10
```

并行运行并保存 JSON/CSV 产物：

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

汇总已有结果文件，不重新调用模型：

```bash
python -m src.evaluator \
  --summarize outputs/evaluation/small50_mimic_iii_results_fixed_parallel.json outputs/evaluation/small50_eicu_results_fixed_parallel.json \
  --summary-output outputs/evaluation/small50_grouped_summary.json \
  --summary-csv outputs/evaluation/small50_grouped_summary.csv
```

评估器会统计总体执行成功率、总体执行匹配率、可回答样本执行成功率、可回答样本执行匹配率、不可回答 Precision/Recall/F1、平均修复轮数和模型 JSON/调用错误数。

常用 evaluator 参数：

- `--workers N`：并行评估样本。
- `--example-type auto|basic|time`：默认 `auto`，`*_not_empty.json` 会自动走时间推理 few-shot，`*_empty.json` 会走基础 few-shot。
- `--no-cache`：强制实时 API 调用。
- `--no-heuristics`：关闭确定性模板，用于纯 LLM prompt 对比。
- `--rule-only-answerability`：跳过 LLM 可回答性判断，仅使用规则判断。
- `--rows-csv`：保存单样本明细。
- `--summary-output`、`--summary-csv`：保存分库汇总。

## 自动修复 Demo

运行内置的错误 SQL 修复案例：

```bash
python -m src.repair_demo --json
```

默认输出写入 `outputs/evaluation/repair_cases.json`，单次运行日志写入 `outputs/runs/`。当前 demo 包含错表名、错字段名和未加引号过滤值 3 类案例。

## Web UI

启动可视化界面：

```bash
streamlit run src/app.py
```

页面支持数据库、模型、prompt、查询模式、最大返回行数、SQL 超时和模型缓存开关；也支持从测试集载入样本、查看候选 Schema、运行单条问题、展示 SQL/结果表/修复日志、做单样本 gold SQL 对比，以及运行自动修复演示。

## 测试

```bash
pytest
```

测试覆盖 Schema 检索、SQL 安全校验、只读执行、few-shot 示例、生成模板、可回答性规则、评估指标、LLM 校验逻辑和修复机制。真实数据库相关测试在数据库文件缺失时会自动跳过。

## 数据与复现说明

测试集位于：

```text
data/EHRSQL/测试集/mimic_iii_test_empty.json
data/EHRSQL/测试集/mimic_iii_test_not_empty.json
data/EHRSQL/测试集/eicu_test_empty.json
data/EHRSQL/测试集/eicu_test_not_empty.json
```

few-shot 代表样例位于：

```text
data/EHRSQL/示例数据/mimic_iii_test_split_tag_representatives.json
data/EHRSQL/示例数据/mimic_iii_test_split_impossible_representatives.json
data/EHRSQL/示例数据/eicu_test_split_tag_representatives.json
data/EHRSQL/示例数据/eicu_test_split_impossible_representatives.json
```

本项目只把代码、配置、prompt 和小型 JSON 数据纳入版本控制；`.env`、数据库文件、缓存、运行日志和评估产物均作为本地复现材料保存。
