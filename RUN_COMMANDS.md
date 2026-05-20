# Run Commands

This file is a compact command checklist for reproducing the main experiments.
For the longer experiment ledger and result provenance, see `PLAN.md`.

Run commands from the repository root.

## Environment

```bash
export PYTHON_BIN=python

export GEMMA_MODEL_PATH=/path/to/google/gemma-2-2b
export GEMMA9B_MODEL_PATH=/path/to/google/gemma-2-9b
export QWEN3_4B_MODEL_PATH=/path/to/Qwen3-4B-Base
export DOLA_CUSTOM_GENERATE=$PWD/third_party/transformers-community-dola/custom_generate/generate.py

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Step 3 requires an annotation API key:

```bash
export GPT_API_KEY=<your_api_key>
```

## Data Construction

```bash
$PYTHON_BIN scripts/step1_extract_queries.py
$PYTHON_BIN scripts/step2_retrieve.py
$PYTHON_BIN scripts/step3_annotate.py
$PYTHON_BIN scripts/step4_build_dataset.py
$PYTHON_BIN scripts/step4b_build_retrieved_dataset.py
$PYTHON_BIN scripts/step5_prepare_test.py
```

Expected outputs:

```text
data/final/train.json
data/final/eval.json
data/final_retrieved/train.json
data/final_retrieved/eval.json
data/final/test_retrieval_nq.json
data/final/test_retrieval_triviaqa.json
```

## Hidden-State Extraction

Gold-vs-distractor setting:

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN scripts/step6_extract_hidden_states.py --model gemma2b --batch_size 32
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN scripts/step6_extract_hidden_states.py --model qwen3_4b --batch_size 4
```

Retrieved-doc setting:

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN scripts/step6_extract_hidden_states_retrieved.py --model gemma2b --batch_size 32
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN scripts/step6_extract_hidden_states_retrieved.py --model qwen3_4b --batch_size 4
```

Use `--model all` when all model paths are configured.

## Probe Baselines

```bash
$PYTHON_BIN scripts/step8_train_probe.py --model all --classifier all
$PYTHON_BIN scripts/step8_train_probe_retrieved.py --model all --classifier all
$PYTHON_BIN scripts/step8b_dpr_baseline.py
$PYTHON_BIN scripts/step8b_dpr_baseline_retrieved.py
```

Optional external evaluator baselines:

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN scripts/step8_crag_baseline.py --mode retrieved
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN scripts/step8_selfrag_baseline.py --mode retrieved --model_name /path/to/selfrag_llama2_13b
```

## CSV Training

Gold-vs-distractor CSV:

```bash
bash scripts/run_step9_dual_gpu.sh
bash scripts/run_step9_gemma9b.sh
bash scripts/run_step9_qwen3_4b.sh
$PYTHON_BIN scripts/step9b_reeval_per_subset.py --model all
```

Retrieved-doc CSV:

```bash
bash scripts/run_step9r_gemma2b.sh
bash scripts/run_step9r_gemma2b_cls18_inject_sweep.sh
bash scripts/run_step9r_qwen3_4b.sh
bash scripts/run_step9r_qwen3_4b_cls22_hparam_grid.sh
```

Recommended retrieved-doc checkpoints:

```text
Gemma2-2B: inject_14_cls_18
Qwen3-4B: inject_21_cls_22
```

## Logit Lens

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN scripts/step10_logit_lens.py --model gemma2b
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN scripts/step10_logit_lens.py --model qwen3_4b
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN scripts/step10_logit_lens_retrieved.py --model gemma2b --dataset nq
$PYTHON_BIN scripts/step10b_plot_logit_lens.py
```

## Pipeline Generation

Baselines:

```bash
CAD_ALPHA=0.5 SKIP_EXISTING=1 bash scripts/run_step11_greedy_baselines_with_eval.sh gemma2b all 0
CAD_ALPHA=0.5 SKIP_EXISTING=1 bash scripts/run_step11_greedy_baselines_with_eval.sh qwen3_4b all 0
```

CSV-gated methods:

```bash
for METHOD in csv_gated_vanilla_rag csv_gated_cad csv_gated_acd csv_gated_context_ucd csv_gated_dola; do
  METHOD="$METHOD" ALPHAS="0.5" TAUS="0.5 0.4 0.3" SKIP_EXISTING=1 \
    bash scripts/run_step11_cad_alpha_sweep_with_eval.sh gemma2b all 0
done

for METHOD in csv_gated_vanilla_rag csv_gated_cad csv_gated_acd csv_gated_context_ucd csv_gated_dola; do
  METHOD="$METHOD" ALPHAS="0.5" TAUS="0.5 0.4 0.3" SKIP_EXISTING=1 \
    bash scripts/run_step11_cad_alpha_sweep_with_eval.sh qwen3_4b all 0
done
```

Single-file example:

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN scripts/step11_contrastive_decoding.py \
  --model gemma2b \
  --dataset nq \
  --method csv_gated_cad \
  --alpha 0.5 \
  --tau 0.5 \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 20
```

## Evaluation and Collection

```bash
$PYTHON_BIN scripts/step12_evaluate_pipeline.py --all
$PYTHON_BIN scripts/collect_finished_results.py
```

Main outputs:

```text
results/pipeline/
results/pipeline/pipeline_eval_summary.csv
results/pipeline/pipeline_eval_summary.json
results/all_finished_experiments.csv
```

## Analysis Utilities

```bash
$PYTHON_BIN scripts/analyze_pipeline_paper_supplements.py
$PYTHON_BIN scripts/analyze_csv_gate_gain.py
$PYTHON_BIN scripts/analyze_csvr_layer_hparams.py
$PYTHON_BIN scripts/collect_step9r_hparam_studies.py
```
