# TODO：实验三当前后续任务

当前基础链路已经实现并通过 6 条 smoke 样本验证。旧 TODO 中的工程决策已汇入 `实验三实验计划.md`，对应功能也已基本落地；本文件只保留下一阶段仍需要执行和验收的任务。

## 已完成基线

- 支持 `mimic_iii` 和 `eicu` 两个 SQLite 数据库。
- 完成 Schema 加载、Schema 检索、SQL 安全校验、只读执行。
- 完成 DeepSeek API 客户端、Agent 编排、CLI、evaluator、运行日志。
- 完成 Streamlit Web 可视化界面。
- `.env` 保存真实 API key，`.env.example` 保持占位值。
- 单元测试通过：`21 passed`。
- 6 条 smoke 样本最终结果：
  - MIMIC-III：3/3 匹配。
  - eICU：3/3 匹配。

## P0：扩大评估样本

目标：从 smoke 验证扩展到可写入报告的小规模指标，并记录已用样本，后续扩大测试时避免重复抽样。

当前已完成：

- 每库 20 条小样本评估：
  - 样本文件：
    - `outputs/evaluation/small20_mimic_iii_samples.json`
    - `outputs/evaluation/small20_eicu_samples.json`
  - 结果文件：
    - `outputs/evaluation/small20_mimic_iii_results_fixed_parallel.json`
    - `outputs/evaluation/small20_eicu_results_fixed_parallel.json`
    - `outputs/evaluation/small20_summary_fixed_parallel.json`
  - 规模：每库 15 条可回答 + 5 条不可回答，共 40 条。
  - 结果：40/40 匹配，30/30 可回答样本执行匹配，10/10 不可回答识别正确。
- 每库 50 条小样本评估：
  - 样本文件：
    - `outputs/evaluation/small50_mimic_iii_samples.json`
    - `outputs/evaluation/small50_eicu_samples.json`
  - 结果文件：
    - `outputs/evaluation/small50_mimic_iii_results_fixed_parallel.json`
    - `outputs/evaluation/small50_eicu_results_fixed_parallel.json`
    - `outputs/evaluation/small50_summary_fixed_parallel.json`
  - 规模：每库 40 条可回答 + 10 条不可回答，共 100 条。
  - 结果：100/100 匹配，80/80 可回答样本执行匹配，20/20 不可回答识别正确。
  - 并行方式：`src.evaluator --workers 6` 按样本并行。

已使用样本范围：

- `small20` 是 `small50` 的子集，后续扩大测试只需要排除 `small50_*_samples.json` 中的样本 `id`。
- MIMIC-III `small50`：
  - `data/EHRSQL/测试集/mimic_iii_test_empty.json` 中前 25 条可回答样本。
  - `data/EHRSQL/测试集/mimic_iii_test_not_empty.json` 中前 15 条可回答样本。
  - `data/EHRSQL/测试集/mimic_iii_test_empty.json` 中 impossible 起始段 10 条，索引 `135-144`。
- eICU `small50`：
  - `data/EHRSQL/测试集/eicu_test_empty.json` 中前 25 条可回答样本。
  - `data/EHRSQL/测试集/eicu_test_not_empty.json` 中前 15 条可回答样本。
  - `data/EHRSQL/测试集/eicu_test_empty.json` 中 impossible 起始段 10 条，索引 `122-131`。

后续扩大测试约束：

- 生成每库 100 条或更多样本时，先读取并排除：
  - `outputs/evaluation/small50_mimic_iii_samples.json`
  - `outputs/evaluation/small50_eicu_samples.json`
- 按 `id` 去重，不按问题文本去重。
- 优先按 `tag` 分层抽样，避免继续只覆盖前几个模板。
- 新样本文件建议命名为 `outputs/evaluation/small100_<db>_samples.json` 或 `outputs/evaluation/stratified100_<db>_samples.json`。

验收标准：

- 下一轮扩大测试不得重复使用上述 100 条已评估样本。
- 得到分库执行成功率、执行匹配率、不可回答 Precision/Recall/F1。

## P1：补 evaluator 指标（已完成）

目标：让评估结果能直接用于实验报告。

- 已增加不可回答判断指标：
  - Precision
  - Recall
  - F1
- 已区分可回答样本执行成功率和总体执行成功率。
- 已统计平均修复轮数。
- 已统计模型格式错误数。
- 已支持按数据库分组的汇总表：
  - `--summarize <result.json>...`
  - `--summary-output <summary.json>`
  - `--summary-csv <summary.csv>`
- 已支持单样本明细 CSV：
  - `--rows-csv <rows.csv>`
- 当前已生成：
  - `outputs/evaluation/small50_grouped_summary.json`
  - `outputs/evaluation/small50_grouped_summary.csv`

验收标准：

- `outputs/evaluation/*.json` 中包含完整指标。
- 报告可以直接引用 evaluator 输出。
- 单元测试覆盖指标计算、分库汇总和 CSV 输出。

## P1：错误分析与规则增强（已完成）

目标：针对小样本评估中真实出现的高频错误做最小修复。

已定位的主要失败：

- Schema 召回不足：
  - `small50_mimic_iii_results_parallel.json` 中 cost/lab/drug 问题大量被误判不可回答，例如 `What is the price of albumin?`、`What is the CD3 %?`。
  - `small50_eicu_results_parallel.json` 中 treatment/lab/medication cost 问题大量被误判不可回答，例如 `What is the cost of bronchodilator - beta-agonist?`、`Can you tell me the costs of Klonopin?`。
- 实体值匹配过宽或过窄：
  - `What are the methods of intake for the chemo syringe (chemo)?` 需要匹配库内实体 `syringe (chemo)`。
  - `What is the price of a drug named amoxicillin oral susp. suspension?` 需要匹配库内实体 `amoxicillin oral susp.`。
  - `How much does it cost for folate lab tests?` 需要去掉 `lab tests` 后缀并走 eICU `lab` 事件表。
- SQL 可执行但结果不等价：
  - `What is the cost of nitroglycerin - oral?` 曾生成 medication cost SQL，但 gold 使用 eICU `treatment` 事件。
- 时间推理口径不一致：
  - `Could you provide the length of stay for patient 75581's initial hospital admission?` 中 `initial` 需要按 first admission 处理。

已完成修复：

- Schema 检索：
  - cost/price/charge 触发 `COST` 表召回。
  - cost 类问题同时提升 procedure/treatment/lab/drug 事件表召回。
- 可回答性规则：
  - cost 问题只要 schema 中存在 `COST` 和事件表，即判为可回答，避免误拒绝。
  - 高频 impossible 模板（医生、陪护电话、家族史、用药限制/作用、出院材料、外院/其他科室用药、未来就诊）直接判为不可回答。
- 生成规则：
  - 将 cost 模板拆为 procedure/lab/drug/diagnosis 四类。
  - 使用库内真实实体做精确归一化，避免模型或字符串模板误选事件类型。
  - 将 `initial hospital admission` 纳入 first hospital stay 口径。
- 单元测试：
  - 覆盖 Schema cost 召回。
  - 覆盖 cost 可回答性规则。
  - 覆盖 MIMIC-III procedure/lab/drug cost、eICU treatment/lab/medication cost、药品名归一化。

当前回归结果：

- `outputs/evaluation/small50_mimic_iii_results_fixed_parallel.json`：50/50 匹配。
- `outputs/evaluation/small50_eicu_results_fixed_parallel.json`：50/50 匹配。
- `outputs/evaluation/small50_summary_fixed_parallel.json`：100/100 匹配，80/80 可回答样本执行匹配，20/20 不可回答识别正确。
- 离线规则回归（禁用 LLM）：
  - `outputs/evaluation/small50_summary_p1_verify.json`：100/100 匹配，`json_or_model_errors = 0`。

验收标准：

- 每个修复都能对应到明确失败案例。（已完成）
- 修复后重跑小样本，结果不回退。（已完成）

## P1：验证自动修复机制（已完成）

目标：准备报告和演示中的“自动纠错与反思”案例。

- 已构造 3 条错误 SQL：
  - 错字段名。
  - 错表名。
  - 过滤条件导致 SQLite 报错。
- 已让 Agent 根据 validation 或 SQLite error 调用 repair：
  - 错表名：`unknown table(s): admission` -> `ADMISSIONS`。
  - 错字段名：`no such column: admissions.discharge_time` -> `ADMISSIONS.DISCHTIME`。
  - 过滤条件：`no such column: f` -> `patients.gender = 'f'`。
- 已保存修复前 SQL、错误信息、修复后 SQL、最终结果：
  - `outputs/evaluation/repair_cases.json`
- Web 页面已增加“自动修复演示”面板，运行后沿用现有结果页展示 `repairs` 修复日志。
- 新增单元测试：
  - `tests/test_repair_mechanism.py`

验收标准：

- 至少 1 个自动修复案例可稳定复现。（已完成，3/3 成功）
- Web 页面能展示修复日志。（已完成）

## P2：Web UI 复查

目标：确保最终演示体验稳定。

- 用 6 条 smoke 样本作为固定演示样例。
- 检查数据库、模型、prompt、查询模式切换。
- 检查测试样本载入和 gold SQL 对比。
- 检查不可回答样本展示。
- 检查追问模式上下文不会影响“新查询”。
- 截取典型页面截图保存到 `outputs/screenshots/`。

验收标准：

- 浏览器中能稳定完成普通查询、时间推理、不可回答三类演示。

## P2：报告与视频材料

目标：形成最终提交材料。

- 整理系统架构图或模块说明。
- 固定报告中的案例：
  - 普通可回答：药物给药途径。
  - 时间推理：住院时长 first/last。
  - 不可回答：未来就诊计划或医疗建议。
  - 自动修复：字段/表错误修复。
- 汇总 prompt 对比或至少说明当前 `schema` prompt 的作用。
- 写明环境复现方式：
  - 创建 `.venv`
  - 安装 `requirements.txt`
  - 配置 `.env`
  - 运行 CLI / Streamlit
- 检查提交物不包含 `.env`、`.venv`、缓存和真实 API key。

验收标准：

- 报告、代码、演示视频材料齐备。
