from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.planner import build_followup_context, run_agent
from src.evaluator import results_equal
from src.repair_demo import REPAIR_CASES
from src.schema_index import format_schema_context, retrieve_schema
from src.sql_executor import execute_sql


EHRSQL_DATA_DIR = ROOT_DIR / "data" / "EHRSQL"
TEST_FILES = {
    "mimic_iii": {
        "基础集": EHRSQL_DATA_DIR / "测试集" / "mimic_iii_test_empty.json",
        "时间推理集": EHRSQL_DATA_DIR / "测试集" / "mimic_iii_test_not_empty.json",
    },
    "eicu": {
        "基础集": EHRSQL_DATA_DIR / "测试集" / "eicu_test_empty.json",
        "时间推理集": EHRSQL_DATA_DIR / "测试集" / "eicu_test_not_empty.json",
    },
}


def main() -> None:
    st.set_page_config(page_title="EHRSQL Agent", layout="wide")
    _init_state()

    st.title("EHRSQL Text-to-SQL Agent")

    with st.sidebar:
        db_id = st.selectbox("数据库", ["mimic_iii", "eicu"], index=0)
        model = st.selectbox("模型", ["deepseek-v4-flash", "deepseek-v4-pro"], index=0)
        prompt_version = st.selectbox("Prompt", ["fewshot", "schema", "base", "reflection"], index=0)
        mode_label = st.radio("查询模式", ["新查询", "基于历史摘要追问"], index=0)
        mode = "followup" if mode_label == "基于历史摘要追问" else "new_query"
        max_rows = st.number_input("最大返回行数", min_value=1, max_value=1000, value=100, step=25)
        timeout_seconds = st.number_input("SQL 超时秒数", min_value=1.0, max_value=30.0, value=5.0, step=1.0)
        use_cache = st.toggle("使用模型缓存", value=True)

        st.divider()
        st.caption(f"历史轮次：{len(st.session_state.turn_history)}")
        if st.button("清空上下文", use_container_width=True):
            st.session_state.turn_history = []
            st.session_state.selected_sample = None
            st.rerun()

    sample = _sample_picker(db_id)
    default_question = sample.get("question", "") if sample else ""
    question = st.text_area("问题", value=st.session_state.get("question_text", default_question), height=110)
    st.session_state.question_text = question
    example_type = _example_type_from_split(st.session_state.get("active_split_name", "")) if sample else "auto"
    sample_metadata = _sample_prompt_metadata(sample, example_type)

    cols = st.columns([1, 1, 4])
    run_clicked = cols[0].button("运行", type="primary", use_container_width=True)
    schema_clicked = cols[1].button("查看 Schema", use_container_width=True)

    if schema_clicked and question.strip():
        _show_schema(question, db_id)

    if run_clicked:
        if not question.strip():
            st.warning("请输入问题。")
            return
        followup_context = build_followup_context(st.session_state.turn_history) if mode == "followup" else ""
        with st.spinner("Agent running..."):
            result = run_agent(
                question.strip(),
                db_id=db_id,
                model=model,
                prompt_version=prompt_version,
                mode=mode,
                followup_context=followup_context,
                max_rows=int(max_rows),
                timeout_seconds=float(timeout_seconds),
                use_cache=use_cache,
                example_type=example_type,
                sample_metadata=sample_metadata,
            )
        st.session_state.last_result = result
        if result.get("generated_sql") and result.get("execution", {}).get("ok"):
            _append_history(result)

    _repair_demo_panel(db_id, model, prompt_version, int(max_rows), float(timeout_seconds), use_cache)

    if st.session_state.last_result:
        _render_result(st.session_state.last_result, sample)

    _render_history()


def _init_state() -> None:
    st.session_state.setdefault("turn_history", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("selected_sample", None)
    st.session_state.setdefault("question_text", "")
    st.session_state.setdefault("active_split_name", "基础集")


def _sample_picker(db_id: str) -> Optional[Dict[str, Any]]:
    with st.expander("测试集样本", expanded=False):
        split_name = st.selectbox("数据集", list(TEST_FILES[db_id].keys()), index=0)
        st.session_state.active_split_name = split_name
        samples = _load_samples(TEST_FILES[db_id][split_name])
        col1, col2, col3 = st.columns([1, 1, 2])
        if col1.button("随机样本", use_container_width=True):
            st.session_state.selected_sample = random.choice(samples)
            st.session_state.question_text = st.session_state.selected_sample.get("question", "")
            st.rerun()
        show_answerable = col2.toggle("只看可回答", value=False)
        candidates = [item for item in samples if not _is_unanswerable(item)] if show_answerable else samples
        sample_id = col3.selectbox(
            "样本 ID",
            [item.get("id", "") for item in candidates[:300]],
            index=0 if candidates else None,
        )
        if sample_id:
            selected = next((item for item in candidates if item.get("id") == sample_id), None)
            if st.button("载入所选样本", use_container_width=True):
                st.session_state.selected_sample = selected
                st.session_state.question_text = selected.get("question", "") if selected else ""
                st.rerun()

        sample = st.session_state.selected_sample
        if sample and sample.get("db_id") == db_id:
            st.json(
                {
                    "id": sample.get("id"),
                    "is_impossible": sample.get("is_impossible"),
                    "query": sample.get("query"),
                    "t_tag": sample.get("t_tag"),
                },
                expanded=False,
            )
            return sample
    return None


@st.cache_data(show_spinner=False)
def _load_samples(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _show_schema(question: str, db_id: str) -> None:
    retrieved = retrieve_schema(question, db_id)
    with st.expander("候选 Schema", expanded=True):
        st.code(format_schema_context(retrieved), language="text")


def _repair_demo_panel(
    db_id: str,
    model: str,
    prompt_version: str,
    max_rows: int,
    timeout_seconds: float,
    use_cache: bool,
) -> None:
    with st.expander("自动修复演示", expanded=False):
        cases = [case for case in REPAIR_CASES if case["db_id"] == db_id]
        if not cases:
            st.write("当前数据库暂无修复案例。")
            return
        case_ids = [case["id"] for case in cases]
        selected_id = st.selectbox(
            "案例",
            case_ids,
            format_func=lambda case_id: next(case["title"] for case in cases if case["id"] == case_id),
        )
        selected = next(case for case in cases if case["id"] == selected_id)
        st.write(selected["error_type"])
        st.code(selected["broken_sql"], language="sql")
        if st.button("运行修复案例", use_container_width=True):
            with st.spinner("Repair running..."):
                result = run_agent(
                    selected["question"],
                    db_id=db_id,
                    model=model,
                    prompt_version=prompt_version,
                    max_rows=max_rows,
                    timeout_seconds=timeout_seconds,
                    use_cache=use_cache,
                    initial_sql=selected["broken_sql"],
                )
            st.session_state.question_text = selected["question"]
            st.session_state.selected_sample = None
            st.session_state.last_result = result
            st.rerun()


def _render_result(result: Dict[str, Any], sample: Optional[Dict[str, Any]]) -> None:
    st.subheader("运行结果")
    metric_cols = st.columns(4)
    execution = result.get("execution") or {}
    validation = result.get("validation") or {}
    metric_cols[0].metric("SQL 校验", "通过" if validation.get("ok") else "未通过")
    metric_cols[1].metric("执行状态", "成功" if execution.get("ok") else "未执行/失败")
    metric_cols[2].metric("返回行数", execution.get("row_count", 0))
    metric_cols[3].metric("耗时 ms", result.get("timing", {}).get("total_elapsed_ms", 0))

    st.info(result.get("final_answer", ""))

    tab_steps, tab_sql, tab_result, tab_eval, tab_log = st.tabs(["步骤", "SQL", "结果表", "单条评估", "日志"])
    with tab_steps:
        _render_steps(result)
    with tab_sql:
        sql = result.get("generated_sql") or ""
        if sql:
            st.code(sql, language="sql")
        else:
            st.write("未生成 SQL。")
        if result.get("repairs"):
            st.write("修复日志")
            for repair in result["repairs"]:
                st.json(repair, expanded=False)
    with tab_result:
        _render_execution_table(execution)
    with tab_eval:
        _render_sample_eval(result, sample)
    with tab_log:
        st.json(result, expanded=False)


def _render_steps(result: Dict[str, Any]) -> None:
    schema = result.get("schema_candidates") or {}
    tables = schema.get("tables") or []
    st.write("候选表")
    if tables:
        schema_rows = [
            {
                "table": table.get("name"),
                "score": table.get("score"),
                "rows": table.get("row_count"),
                "matched_terms": ", ".join(table.get("matched_terms") or []),
            }
            for table in tables
        ]
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)
        with st.expander("候选字段"):
            for table in tables:
                column_names = [col.get("name") for col in table.get("columns", [])]
                st.write(f"{table.get('name')}: {', '.join(column_names)}")
    else:
        st.write("无候选表。")

    st.write("可回答性判断")
    st.json(result.get("answerability") or {}, expanded=False)

    st.write("SQL 校验")
    st.json(result.get("validation") or {}, expanded=False)

    if result.get("errors"):
        st.write("错误")
        st.json(result["errors"], expanded=False)
    if result.get("fewshot"):
        st.write("Few-shot")
        st.json(result["fewshot"], expanded=False)


def _render_execution_table(execution: Dict[str, Any]) -> None:
    if not execution:
        st.write("未执行。")
        return
    if not execution.get("ok"):
        st.error(_error_message(execution))
        return
    rows = execution.get("rows") or []
    columns = execution.get("columns") or []
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=columns), use_container_width=True, hide_index=True)
    else:
        st.write("执行成功，结果为空。")
    if execution.get("truncated"):
        st.warning("结果已按最大返回行数截断。")


def _render_sample_eval(result: Dict[str, Any], sample: Optional[Dict[str, Any]]) -> None:
    if not sample:
        st.write("未载入测试样本。")
        return
    if _is_unanswerable(sample):
        predicted_refusal = not result.get("generated_sql") or result.get("answerability", {}).get("answerable") is False
        st.metric("不可回答判断", "正确" if predicted_refusal else "错误")
        return

    gold_sql = sample.get("query")
    if not gold_sql:
        st.write("样本没有 gold SQL。")
        return
    gold_execution = execute_sql(gold_sql, db_id=sample.get("db_id", result.get("db_id", "mimic_iii")))
    predicted_execution = result.get("execution") or {}
    if not gold_execution.get("ok"):
        st.error(f"Gold SQL 执行失败：{_error_message(gold_execution)}")
        return
    if not predicted_execution.get("ok"):
        st.error("预测 SQL 未成功执行。")
        return
    matched = results_equal(gold_execution, predicted_execution, order_sensitive="order by" in gold_sql.lower())
    st.metric("执行结果比较", "一致" if matched else "不一致")
    with st.expander("Gold SQL"):
        st.code(gold_sql, language="sql")
        _render_execution_table(gold_execution)


def _render_history() -> None:
    with st.expander("追问上下文", expanded=False):
        history = st.session_state.turn_history
        if not history:
            st.write("当前无历史摘要。")
            return
        st.code(build_followup_context(history), language="text")


def _append_history(result: Dict[str, Any]) -> None:
    execution = result.get("execution") or {}
    schema = result.get("schema_candidates") or {}
    used_tables = [table.get("name") for table in schema.get("tables", [])[:4]]
    st.session_state.turn_history.append(
        {
            "turn_id": len(st.session_state.turn_history) + 1,
            "db_id": result.get("db_id"),
            "question": result.get("question"),
            "sql": result.get("generated_sql"),
            "result_summary": result.get("final_answer"),
            "used_tables": used_tables,
            "key_entities": [],
            "row_count": execution.get("row_count", 0),
        }
    )


def _is_unanswerable(sample: Dict[str, Any]) -> bool:
    query = sample.get("query")
    return sample.get("is_impossible") is True or query is None or str(query).strip().lower() in {"", "null", "nan", "none"}


def _example_type_from_split(split_name: str) -> str:
    return "time" if split_name == "时间推理集" else "basic"


def _sample_prompt_metadata(sample: Optional[Dict[str, Any]], example_type: str) -> Dict[str, Any]:
    return {
        "example_type": example_type,
        "t_tag": sample.get("t_tag") if sample else [],
        "tag": sample.get("tag") if sample else None,
        "id": sample.get("id") if sample else None,
    }


def _error_message(payload: Dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        return error.get("message") or error.get("type") or "unknown error"
    return str(error)


if __name__ == "__main__":
    main()
