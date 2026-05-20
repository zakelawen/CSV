# RAG-Probe Experiment Ledger

This file is the English experiment ledger for the repository. It records the
main data splits, classifier results, pipeline settings, and source-of-truth
result locations used while preparing the paper and release.

The current project narrative is:

```text
Use a retrieved-document utility classifier as a plug-in gate for existing RAG
and contrastive decoding methods. If the retrieved document is predicted useful,
run the original document-using method. If not, fall back to a no-document prompt
or another conservative strategy.
```

## Current Status

- The project has two classifier tracks and one end-to-end pipeline track.
- The gold-vs-distractor track is a diagnostic setting that tests whether LLM
  hidden states can separate gold evidence from distracting retrieved passages.
- The retrieved-doc track is the main pipeline setting. It predicts whether the
  actual top-1 retrieved document is answer-supporting.
- Gemma2-2B and Qwen3-4B have completed retrieved-doc CSV gates and pipeline
  evaluation.
- Gemma2-9B has completed the gold-vs-distractor CSV track and probe baselines,
  but it is not included in the final pipeline table.

## Source-of-Truth Files

When a metric appears in more than one analysis file, use the following order:

| Quantity | Source file |
|---|---|
| Gold-vs-distractor data size | `data/final/train.json`, `data/final/eval.json` |
| Retrieved-doc data size | `data/final_retrieved/train.json`, `data/final_retrieved/eval.json` |
| Pipeline test data size | `data/final/test_retrieval_nq.json`, `data/final/test_retrieval_triviaqa.json` |
| Probe baselines | `results/probe/`, `results/probe_retrieved/` |
| Gold-vs-distractor CSV | `results/csv/` |
| Retrieved-doc CSV | `results/csv_retrieved/` |
| Qwen3-4B pipeline CSV checkpoint | `results/csv_retrieved_qwen3_4b_best_for_pipeline/` |
| Pipeline generations and evals | `results/pipeline/` or the archived `autodl_5090_2/results/pipeline/` |
| Final pipeline summary | `autodl_5090_2/results/pipeline/pipeline_eval_summary.csv` |
| Paper-ready analysis tables | `results/analysis/` |
| Manual label validation | `results/manual_validation/` |

For the final paper table, the archived pipeline summary contains 84 evaluation
rows: 42 for Gemma2-2B and 42 for Qwen3-4B.

## Dataset Sizes

| Split | Path | Size |
|---|---|---:|
| Gold-vs-distractor train | `data/final/train.json` | 2,940 |
| Gold-vs-distractor eval | `data/final/eval.json` | 736 |
| Retrieved-doc train | `data/final_retrieved/train.json` | 5,880 |
| Retrieved-doc eval | `data/final_retrieved/eval.json` | 2,135 |
| NQ pipeline test | `data/final/test_retrieval_nq.json` | 3,610 |
| TriviaQA pipeline test | `data/final/test_retrieval_triviaqa.json` | 11,313 |

The retrieved-doc training split is balanced by label. The retrieved-doc eval
split keeps the natural label distribution.

## Main Methods

### Probe Baselines

Probe baselines train classifiers on cached last-token hidden states:

- Logistic regression
- MLP
- Nearest centroid
- Mass-mean classifier

These baselines test whether the frozen LLM representation already contains a
document-utility signal.

### Context Separator Vector

The main method learns a Context Separator Vector (CSV). During training, a
vector is injected into an intermediate layer of a frozen LLM:

```text
H_tilde^k = H^k + lambda * v
```

The modified hidden states are passed through the remaining frozen layers. At a
classification layer, the last prompt-token representation is scored against
answer-supporting and distracting prototypes.

### External Evaluator Baselines

The external baselines are:

- DPR retriever score
- CRAG retrieval evaluator
- Self-RAG retrieval evaluator
- Cross-Encoder answer-support classifier
- Low-rank representation head

The Cross-Encoder baseline uses `cross-encoder/ms-marco-MiniLM-L6-v2` with
`answer_mode=none` for the official comparison.

## Gold-vs-Distractor Results

This setting compares known gold evidence against retrieved distracting passages.
It is used as a hidden-state separability diagnostic, not as the Step 11 gate.

| Model | Classifier | Best config | Combined AUROC | Combined accuracy |
|---|---|---|---:|---:|
| External | DPR dot | - | 0.2919 | 0.5000 |
| External | CRAG | - | 0.4519 | 0.5000 |
| External | Self-RAG | - | 0.6451 | 0.6046 |
| External | Cross-Encoder | MiniLM-L6 | 0.9922 | 0.9524 |
| Gemma2-2B | LR | layer 5 | 0.8029 | 0.7446 |
| Gemma2-2B | MLP | layer 5 | 0.8568 | 0.7779 |
| Gemma2-2B | Centroid | layer 5 | 0.7458 | 0.6855 |
| Gemma2-2B | Mass-Mean | layer 5 | 0.7456 | 0.6875 |
| Gemma2-2B | Low-rank Rep. Head | final/r8/a16 | 0.8891 | 0.8152 |
| Gemma2-2B | CSV | inject 1, cls 16 | 0.9964 | 0.9592 |
| Gemma2-9B | LR | layer 29 | 0.7605 | 0.7052 |
| Gemma2-9B | MLP | layer 31 | 0.7702 | 0.7052 |
| Gemma2-9B | Centroid | layer 29 | 0.7415 | 0.6712 |
| Gemma2-9B | Mass-Mean | layer 29 | 0.7404 | 0.6698 |
| Gemma2-9B | CSV | inject 0, cls 30 | 0.9974 | 0.9654 |
| Qwen3-4B | LR | layer 23 | 0.8127 | 0.7486 |
| Qwen3-4B | MLP | layer 23 | 0.8325 | 0.7636 |
| Qwen3-4B | Centroid | layer 23 | 0.7768 | 0.7154 |
| Qwen3-4B | Mass-Mean | layer 23 | 0.7777 | 0.7154 |
| Qwen3-4B | Low-rank Rep. Head | final/r8/a16 | 0.9309 | 0.8655 |
| Qwen3-4B | CSV | inject 6, cls 24 | 0.9961 | 0.9552 |

The gold-vs-distractor CSV results show that a lightweight steering vector can
strongly amplify a hidden-state document-quality signal.

## Retrieved-Document Results

This is the main utility-classification setting for the pipeline. It asks
whether the real top-1 retrieved document is answer-supporting.

| Model | Classifier | Best config | Combined AUROC | Combined accuracy |
|---|---|---|---:|---:|
| External | DPR retriever score | - | 0.6376 | 0.6253 |
| External | CRAG | - | 0.5171 | 0.4244 |
| External | Self-RAG | - | 0.7688 | 0.6984 |
| External | Cross-Encoder | MiniLM-L6 | 0.7985 | 0.7504 |
| Gemma2-2B | LR | layer 18 | 0.8497 | 0.7536 |
| Gemma2-2B | MLP | layer 18 | 0.8486 | 0.7607 |
| Gemma2-2B | Centroid | layer 18 | 0.8403 | 0.7696 |
| Gemma2-2B | Mass-Mean | layer 18 | 0.8403 | 0.7756 |
| Gemma2-2B | Low-rank Rep. Head | final/r8/a16 | 0.8284 | 0.7494 |
| Gemma2-2B | CSV_R | inject 14, cls 18 | 0.9029 | 0.8375 |
| Gemma2-9B | LR | layer 27 | 0.8893 | 0.8005 |
| Gemma2-9B | MLP | layer 29 | 0.8995 | 0.8108 |
| Gemma2-9B | Centroid | layer 27 | 0.8780 | 0.8108 |
| Gemma2-9B | Mass-Mean | layer 27 | 0.8781 | 0.8173 |
| Qwen3-4B | LR | layer 22 | 0.8918 | 0.8262 |
| Qwen3-4B | CSV_R | inject 21, cls 22 | 0.9159 | 0.8553 |

The retrieved-doc setting is harder and closer to the real RAG pipeline. CSV_R
is the classifier used by the Step 11 gate.

## Main Retrieved-Doc CSV Checkpoints

| Model | Injection layer | Classification layer | AUROC | Accuracy | Best epoch |
|---|---:|---:|---:|---:|---:|
| Gemma2-2B | 14 | 18 | 0.902907 | 0.837471 | 7 |
| Qwen3-4B | 21 | 22 | 0.915867 | 0.855269 | 20 |

Default training settings:

```text
lambda = 5.0
cosine prototype temperature = 0.1
seed = 42
```

## Additional Baselines

### Cross-Encoder

Script:

```text
scripts/step15a_train_answer_support_cross_encoder.py
```

Official setting:

```text
model = cross-encoder/ms-marco-MiniLM-L6-v2
answer_mode = none
epochs = 3
batch_size = 16
learning_rate = 2e-5
```

The cross-encoder is a strong external pairwise relevance classifier. It does
not test whether the target LLM's hidden states naturally encode document
utility.

### Low-Rank Representation Head

Script:

```text
scripts/step15b_train_lowrank_rep_head.py
```

Main setting:

```text
layer = -1
rank = 8
alpha = 16
batch_size = 256
learning_rate = 1e-3
max_epochs = 30
early_stopping_patience = 5
```

This baseline trains a small low-rank residual head on cached frozen hidden
states. It is inspired by parameter-efficient representation adaptation, but it
is not full LoRA or full ReFT because it does not modify the LLM forward pass.

## Logit Lens Diagnostics

Main script:

```text
scripts/step10_logit_lens.py
```

Retrieved-doc script:

```text
scripts/step10_logit_lens_retrieved.py
```

The diagnostic compares answer likelihood under:

```text
no_ctx
gold_ctx
distracting_ctx
```

The completed runs show the expected pattern:

```text
gold_ctx > no_ctx approximately >= distracting_ctx
```

For Gemma2-2B on retrieved-doc NQ, the answer-signal peak for relevant retrieved
documents remains around layer 23.

## End-to-End Pipeline

The pipeline compares six baseline methods:

```text
no_doc
vanilla_rag
cad_fixed
acd
context_ucd
dola
```

and five CSV-gated plug-in methods:

```text
csv_gated_vanilla_rag
csv_gated_cad
csv_gated_acd
csv_gated_context_ucd
csv_gated_dola
```

The gate logic is:

```text
if P(relevant) >= tau:
    run the original document-using baseline
else:
    fall back to no_doc
```

Main settings:

```text
CAD alpha = 0.5
tau values = 0.5, 0.4, 0.3
temperature = 0.0
max_input_length = 512 for the main archived runs
```

The final archived pipeline table contains:

```text
Gemma2-2B: 2 datasets x (6 baselines + 5 gated methods x 3 tau) = 42 rows
Qwen3-4B: 2 datasets x (6 baselines + 5 gated methods x 3 tau) = 42 rows
```

## Pipeline Result Summary

The strongest non-gated baselines in the final summary are:

| Model | Dataset | Best baseline | EM | F1 |
|---|---|---|---:|---:|
| Gemma2-2B | NQ | vanilla_rag | 0.2576 | 0.3647 |
| Gemma2-2B | TriviaQA | acd | 0.4991 | 0.5760 |
| Qwen3-4B | NQ | acd | 0.2806 | 0.3712 |
| Qwen3-4B | TriviaQA | vanilla_rag | 0.3750 | 0.4667 |

For Gemma2-2B with fixed `alpha=0.5` and `tau=0.5`, the gated methods provide
the clearest gains for methods that are more vulnerable to bad retrieval, such
as `cad_fixed`, `context_ucd`, and the local DoLA path. Gains are not monotonic
for every baseline, especially when the original method is already stable or
when falling back to no-doc loses useful evidence.

## Manual Label Validation

Manual validation files are under:

```text
results/manual_validation/
```

Completed validation:

```text
Gold-vs-distractor distractor labels: 100 samples, 95.0% agreement
Retrieved-doc support labels: 50 samples, 74.0% agreement
Retrieved-doc distractor labels: 50 samples, 92.0% agreement
Retrieved-doc overall: 83.0% agreement
```

## Reproduction Command Map

Dataset construction:

```bash
python scripts/step1_extract_queries.py
python scripts/step2_retrieve.py
python scripts/step3_annotate.py
python scripts/step4_build_dataset.py
python scripts/step4b_build_retrieved_dataset.py
python scripts/step5_prepare_test.py
```

Hidden-state extraction:

```bash
python scripts/step6_extract_hidden_states.py --model all
python scripts/step6_extract_hidden_states_retrieved.py --model all
```

Probe baselines:

```bash
python scripts/step8_train_probe.py --model all --classifier all
python scripts/step8_train_probe_retrieved.py --model all --classifier all
python scripts/step8b_dpr_baseline.py
python scripts/step8b_dpr_baseline_retrieved.py
```

Gold-vs-distractor CSV:

```bash
bash scripts/run_step9_dual_gpu.sh
bash scripts/run_step9_gemma9b.sh
bash scripts/run_step9_qwen3_4b.sh
python scripts/step9b_reeval_per_subset.py --model all
```

Retrieved-doc CSV:

```bash
bash scripts/run_step9r_gemma2b.sh
bash scripts/run_step9r_gemma2b_cls18_inject_sweep.sh
bash scripts/run_step9r_qwen3_4b.sh
bash scripts/run_step9r_qwen3_4b_cls22_hparam_grid.sh
```

Pipeline generation and evaluation:

```bash
CAD_ALPHA=0.5 SKIP_EXISTING=1 bash scripts/run_step11_greedy_baselines_with_eval.sh gemma2b all 0
CAD_ALPHA=0.5 SKIP_EXISTING=1 bash scripts/run_step11_greedy_baselines_with_eval.sh qwen3_4b all 0

for METHOD in csv_gated_vanilla_rag csv_gated_cad csv_gated_acd csv_gated_context_ucd csv_gated_dola; do
  METHOD="$METHOD" ALPHAS="0.5" TAUS="0.5 0.4 0.3" SKIP_EXISTING=1 \
    bash scripts/run_step11_cad_alpha_sweep_with_eval.sh gemma2b all 0
done

python scripts/step12_evaluate_pipeline.py --all
python scripts/collect_finished_results.py
```

See `RUN_COMMANDS.md` for a compact command-only checklist.

## Current Paper Narrative

The core claim is not that CSV replaces every reranker. Instead:

- LLM hidden states contain a useful document-utility signal.
- CSV can amplify this signal with a lightweight frozen-model intervention.
- The retrieved-doc CSV gate can be plugged into existing RAG and contrastive
  decoding methods.
- The gate is most useful when the downstream method is sensitive to bad
  retrieval and can safely fall back to no-doc generation.

## Remaining Optional Work

- Add a full Gemma2-9B retrieved-doc CSV_R sweep if the final paper needs a
  third retrieved-doc CSV model.
- Decide whether to include Gemma2-9B in pipeline evaluation.
- Finalize whether the main table reports best-tau results or a fixed
  `tau=0.5` comparison.
