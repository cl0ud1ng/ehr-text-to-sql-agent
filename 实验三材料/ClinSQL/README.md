<p align="center">
  <h1 style="display: inline;">
    Patient-Similarity Cohort Reasoning in Clinical Text-to-SQL
  </h1>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/yifeis02/CLINSQL">🤗 Dataset</a>
  · <a href="https://arxiv.org/abs/2601.09876">📄 Paper</a>
  · <a href="https://github.com/Barryshen1/ClinSQL">💻 GitHub</a>
</p>

## 📰 News
- **2026-01**: ClinSQL has been accepted by EACL 2026 Main!
- **2026-01**: Public release of the CLINSQL paper, dataset, and evaluation code.

## 👋 Overview
![Benchmark Overview](./assets/overview.png)

CLINSQL evaluates large language models on clinical text-to-SQL reasoning over the MIMIC-IV database. Every problem bundles the clinical question, gold BigQuery SQL, reference results, and dual rubric trees used by an LLM judge. The benchmark spans six clinical domains across three difficulty levels, enabling fine-grained analysis of execution success, rubric compliance, and model self-refinement.

CLINSQL is designed to stress-test clinical text-to-SQL systems by providing:

- **633 expert-annotated cohort queries** on MIMIC-IV v3.1 that require patient-similarity cohort construction and multi-step temporal reasoning across heterogeneous EHR tables.
- **Six scenario families with rubric-structured evaluation**, separating critical and non-critical checks, enforcing sequential gating with weighted aggregation, and adding execution-level plausibility checks.

## 🚀 Quickstart
### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Apply for access to the [MIMIC-IV v3.1 dataset](https://physionet.org/content/mimiciv/3.1/) on PhysioNet (training, data use agreement, and credentialing are required). After approval, create a Google Cloud project, enable the BigQuery API, and link the MIMIC-IV public dataset to your project so queries can be billed correctly.

Authenticate to BigQuery via `gcloud auth application-default login` or by exporting `GOOGLE_APPLICATION_CREDENTIALS` with a service account key.

### Run Inference
```bash
bash model_inference_scripts/run_proprietary_models.sh
bash model_inference_scripts/run_vllm_models.sh
```
All generated inference files are saved under `outputs/inference/<model>/<split>/<domain>/<difficulty>/<id>/`.

### Evaluation
```bash
python evaluation/clinical_rubric_scorer.py full <model_name>
```
All evaluation reports are saved under `outputs/evaluation/<model>/<split>/` as:
- `scoring_results.json` (overall summary and per-sample scores)
- `detailed_grading.json` (per-sample rubric transcripts)
- `difficulty_scoring_results.json` (aggregated by difficulty)
- `scenario_scoring_results.json` (aggregated by clinical scenario)

## 📊 Data Card
- **Domains**: Diagnostic Procedures, Disease Diagnosis & Outcomes, Laboratory Results Analysis, Medication Management, Patient Demographics & Admissions, Vital Signs Monitoring.
- **Difficulties**: `easy_level_queries`, `medium_level_queries`, `hard_level_queries` (approximately 3:4:3 ratio per domain).
- **Schema**: BigQuery tables under `physionet-data.mimiciv_3_1_hosp` and `physionet-data.mimiciv_3_1_icu`.
- **Rubrics**: Dual JSON trees (`sql_rubric_tree.json`, `results_rubric_tree.json`) guiding the LLM judge.

## ✍️ Citation
If you use CLINSQL, please cite our paper:
```bibtex
@misc{shen2026patientsimilaritycohortreasoningclinical,
      title={Patient-Similarity Cohort Reasoning in Clinical Text-to-SQL}, 
      author={Yifei Shen and Yilun Zhao and Justice Ou and Tinglin Huang and Arman Cohan},
      year={2026},
      eprint={2601.09876},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.09876}, 
}
```
