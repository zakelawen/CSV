# RAG-Probe 项目完整计划（当前结果更新版 v29）

> 本版本以本仓库内最终结果文件为准；其中 pipeline 最终指标以 `autodl_5090_2/results/pipeline/pipeline_eval_summary.csv` 为准，非 pipeline 分类器/CSV/Logit Lens 指标以 `results/` 与 `data/` 下对应文件为准。当前主线叙事保持为：retrieved-doc classifier 作为 **plugin / gate** 接入已有 RAG 或 decoding baseline，用于决定是否信任 retrieved document、是否启用某个现有方法。
> 更新重点 1：Gemma2-2B 两条 CSV 线已明确区分。旧 gold-vs-distractor Step9 best=`inject_1_cls_16`，AUROC=0.996358，Acc=0.959239；新 retrieved-doc Step9R gate best=`inject_14_cls_18`，AUROC=0.902907，Acc=0.837471，best_epoch=7。
> 更新重点 2：Qwen3-4B retrieved-doc CSV_R 已完成 hparam grid。当前用于 pipeline 的 best=`inject_21_cls_22`，AUROC=0.915867，Acc=0.855269，best_epoch=20；已整理到 `results/csv_retrieved_qwen3_4b_best_for_pipeline/`。
> 更新重点 3：CRAG 与 Self-RAG 两类外部 evaluator baseline 均已同步：gold-vs-distractor 与 retrieved-doc 两个数据口径都有结果。
> 更新重点 4：Gemma2-2B pipeline 已完整同步到本地：6 个原 baseline + 5 个 plugin 方法，plugin tau=`0.5/0.4/0.3`；`results/pipeline/` 保留本地 Gemma2-2B 42 行结果。
> 更新重点 5：最终 pipeline 结果以 5090 回传目录 `autodl_5090_2/results/pipeline/` 为准；6 个 baseline + 5 个 plugin 方法、tau=`0.5/0.4/0.3` 已补齐。该目录的 `pipeline_eval_summary.csv` 当前为 84 行总表，其中 Gemma2-2B 42 行、Qwen3-4B 42 行。
> 更新重点 6：retrieved-doc 版本 Step10R：`scripts/step10_logit_lens_retrieved.py`。Gemma2B/NQ 结果显示 retrieved relevant 的答案信号 peak 仍在 layer 23；当前仅作为解释 classifier-gated pipeline 的诊断证据。
> 更新重点 7：Mass-Mean probe baseline 已补齐 gold-vs-distractor 与 retrieved-doc 两个口径、三模型结果；DPR baseline 已补齐 retrieved-doc 口径，使用 `data/final_retrieved/*` 中保存的 `doc.score` / `retriever_score`，不要与旧 gold-vs-distractor 的 DPR encoder dot baseline 混用。
> 更新重点 8：retrieved-doc 2×2 ablation 已补齐 `(CSV × Prototype)` 四个 cell；新增 `(c) CSV + non-prototype LR head` 结果来自 `c_ablation_retrieved_lr_20260513/results/ablation_csv_nonprototype/`，可直接与普通 LR、Centroid、Ours / CSV_R 对照。
> 更新重点 9：新增两类强基线已整理并写入 0.2/0.3 主分类表：Cross-Encoder answer-support classifier（建议表名 `Cross-Encoder`；gold-vs-distractor + retrieved-doc 两个公平 `answer_mode=none` 设置）和 Low-rank Representation Head（建议表名 `Low-rank Rep. Head`；gold-vs-distractor + retrieved-doc；Gemma2-2B / Qwen3-4B）。汇总见 `results/analysis/additional_baselines_summary.md` / `.csv`。
> 更新重点 10：新增人工标签校验。抽样与标注文件位于 `results/manual_validation/`；已完成 200 条人工标注并汇总到 `results/manual_validation/manual_validation_summary.csv`。Gold-vs-distractor 只校验 distractor label（100 条，Agr.=95.0%）；retrieved-doc 平衡校验 support/distractor（各 50 条），Support Agr.=74.0%，Distractor Agr.=92.0%，Overall Agr.=83.0%。标注网页保留在 `results/manual_validation/annotation_site/`。
> 更新重点 11：新增原始 retrieved-doc 表征重叠可视化，只画 clean / before CSV，不画使用本方法后的表征。输出位于 `results/case_analysis/original_representation_overlap/`，包含 Gemma2-2B / Qwen3-4B × NQ / TriviaQA 共四张图（PNG/SVG/PDF），脚本为 `scripts/plot_original_representation_overlap_retrieved.py`。

---

## 0. 当前结果台账（v29，以本节为准）

### 0.1 结果文件位置

| 结果类型 | 本地路径 | 状态 |
|---|---|---|
| 旧 gold-vs-distractor probe / baseline | `results/probe/` | 已完成；含 LR/MLP/Centroid/Mass-Mean、DPR、CRAG、Self-RAG |
| 旧 gold-vs-distractor CSV | `results/csv/` | 已完成；Gemma2-2B / Gemma2-9B / Qwen3-4B 均有 sweep |
| 新 retrieved-doc probe / baseline | `results/probe_retrieved/` | 已完成；含 LR/MLP/Centroid/Mass-Mean、DPR retriever-score、CRAG、Self-RAG |
| 新 Cross-Encoder answer-support baseline | `results/cross_encoder_gold/`、`results/cross_encoder_retrieved/` | 已完成；正式台账仅保留公平 `answer_mode=none` 的 gold-vs-distractor 与 retrieved-doc 两个设置 |
| 新 Low-rank Representation Head baseline | `results/lowrank_rep_head_gold/`、`results/lowrank_rep_head_retrieved/` | 已完成；Gemma2-2B / Qwen3-4B，`rank=8, alpha=16, layer=-1` |
| 原始表征重叠可视化 | `results/case_analysis/original_representation_overlap/` | 已完成；只画 clean / before CSV，Gemma2-2B / Qwen3-4B × NQ / TriviaQA 共四张图，另有 SVG/PDF 与 metrics CSV |
| 新 retrieved-doc CSV_R | `results/csv_retrieved/` | Gemma2-2B 已完成；Qwen3-4B 原始 sweep 保留 |
| Qwen3-4B pipeline 用 CSV_R best | `results/csv_retrieved_qwen3_4b_best_for_pipeline/` | 已整理；供 Step11 通过 `CSV_RETRIEVED_DIR` 指定 |
| Qwen3-4B CSV_R hparam grid | `results/csv_retrieved_qwen3_4b_cls22_hparam_grid/` | 已完成；当前最佳来自 `lr0p001_ema0p999` |
| Gemma2-2B pipeline 本地镜像 | `results/pipeline/` | 已完成；本地 Gemma2-2B summary 为 42 行，用于交叉检查 |
| Pipeline 最终总表 | `autodl_5090_2/results/pipeline/` | 已完成；最终 summary 总表为 84 行，其中 Gemma2-2B 42 行、Qwen3-4B 42 行；已补评 Qwen3-4B `triviaqa/dola` eval |
| Pipeline 汇总表 | `results/analysis/pipeline_plugin_comparison_by_tau.md` / `.csv` | 已生成；按 tau 分表，Gemma2-2B 与 Qwen3-4B 分上下两部分展示 |
| Pipeline CSV_R validation | `results/pipeline/{model}_retrieved_csv_eval_validation.json` / `autodl_5090_2/results/pipeline/{model}_retrieved_csv_eval_validation.json` | Step11 gated 方法启动时自动生成；用于确认读取的 CSV_R checkpoint 是否正确 |
| Manual label validation | `results/manual_validation/` | 已完成；含抽样文件、简化标注表、网页标注工具、人工完成文件与 agreement 汇总 |
| 5090 上传备份 | `autodl_5090_2/` | 最新完整备份；覆盖 `autodl_5090_1` 的 pipeline/probe/probe_retrieved/csv_retrieved 文件，并额外包含 Qwen pipeline |

### 0.1a 数据来源总索引

本计划中的所有数值表按下列 source-of-truth 回填；若同一指标在旧 analysis 文件和最终 summary 中不一致，**pipeline 指标一律以 `autodl_5090_2/results/pipeline/pipeline_eval_summary.csv` 为准**。表格展示统一保留 4 位小数，`Delta EM / Delta F1` 由未四舍五入的原始 EM / F1 计算后再展示。

| 计划中的数据 | 主要来源文件 | 回填规则 |
|---|---|---|
| 旧 gold-vs-distractor 数据规模 | `data/final/train.json` / `data/final/eval.json` | `train=2940`，`eval=736`；用于旧分类器和旧 CSV 评估 |
| retrieved-doc 数据规模 | `data/final_retrieved/train.json` / `data/final_retrieved/eval.json` | `train=5880`，`eval=2135`；`eval` 为自然 label 分布，`train` 为平衡分布 |
| 人工标签校验 | `results/manual_validation/manual_validation_completed.csv`、`results/manual_validation/manual_validation_summary.csv` | Gold-vs-distractor 只校验 distractor label；retrieved-doc 平衡校验 relevant/distracting；agreement 按 `manual_label == auto_label` 计算 |
| pipeline test 数据规模 | `data/final/test_retrieval_nq.json` / `data/final/test_retrieval_triviaqa.json` | `NQ=3610`，`TriviaQA=11313`；用于 Step11/Step12 EM/F1 |
| 旧 external baseline | `results/probe/dpr_baseline.json`、`results/probe/crag_gold_distractor_baseline_results.json`、`results/probe/selfrag_gold_distractor_baseline_results.json` | 读取 `eval` 指标；DPR 使用 `eval.flat`，CRAG/Self-RAG 使用 `eval` |
| 旧 LR/MLP/Centroid/Mass-Mean probe | `results/probe/{model}_probe_results_{lr,mlp,centroid,mass_mean}.json` | 每个 classifier 按 `combined.auroc` 选择最佳 layer；表中 NQ/TriviaQA/combined 指标取该最佳 layer |
| 旧 Ours / CSV | `results/csv/{model}_csv_per_subset.json`，辅助参考 `results/csv/{model}_csv_sweep.json` | 表中 combined/NQ/TriviaQA 指标取 `results` 字段；best config 取 `str_layer` 与 `cls_layer` |
| retrieved-doc external baseline | `results/probe_retrieved/dpr_baseline.json`、`results/probe_retrieved/crag_baseline_results.json`、`results/probe_retrieved/selfrag_baseline_results.json` | 读取 `eval` 指标；DPR 使用 `eval.flat`，并以保存的 `doc.score` / `retriever_score` 为分数 |
| retrieved-doc LR/MLP/Centroid/Mass-Mean probe | `results/probe_retrieved/{model}_probe_results_{lr,mlp,centroid,mass_mean}.json` | 每个 classifier 按 `combined.auroc` 选择最佳 layer；表中 NQ/TriviaQA/combined 指标取该最佳 layer |
| Cross-Encoder answer-support baseline | `results/cross_encoder_gold/cross-encoder_ms-marco-MiniLM-L6-v2_gold_none/results.json`、`results/cross_encoder_retrieved/cross-encoder_ms-marco-MiniLM-L6-v2_retrieved_none/results.json` 或旧目录 `results/cross_encoder_retrieved/cross-encoder_ms-marco-MiniLM-L6-v2_none/results.json`，汇总见 `results/analysis/additional_baselines_summary.*` | 使用 `answer_mode=none`；gold 口径展开 `relevant_doc/distracting_doc`，retrieved 口径直接读取 `doc/label` |
| Low-rank Representation Head baseline | `results/lowrank_rep_head_gold/{model}_gold_final_rank8_alpha16p0/results.json`、`results/lowrank_rep_head_retrieved/{model}_retrieved_final_rank8_alpha16p0/results.json`，汇总见 `results/analysis/additional_baselines_summary.*` | 冻结 hidden states，final representation 上训练低秩 residual head；gold 读取 `data/hidden_states/`，retrieved 读取 `data/hidden_states_retrieved/` |
| retrieved-doc 2×2 ablation：CSV + non-prototype head | `c_ablation_retrieved_lr_20260513/results/ablation_csv_nonprototype/{model}_retrieved_csv_nonprototype_lr.json`，脚本 `scripts/step14_csv_nonprototype_ablation.py` | 使用各模型 CSV_R best checkpoint 注入 CSV 后，在 CSV-modified hidden states 上训练 LR head；表中取 `results.combined`，NQ/TriviaQA 也在同一 JSON 中 |
| Gemma2-2B retrieved-doc CSV_R | `results/pipeline/gemma2b_retrieved_csv_eval_validation.json`，checkpoint/sweep 见 `results/csv_retrieved/gemma2b_retrieved_csv_sweep.json` | 表中 CSV_R 指标取 validation 的 `computed` 字段；best=`inject_14_cls_18` |
| Qwen3-4B retrieved-doc CSV_R | `results/pipeline/qwen3_4b_retrieved_csv_eval_validation.json`，pipeline checkpoint 见 `results/csv_retrieved_qwen3_4b_best_for_pipeline/`，hparam grid 见 `results/csv_retrieved_qwen3_4b_cls22_hparam_grid/` | 表中 CSV_R 指标取 validation 的 `computed` 字段；best=`inject_21_cls_22`，来自 `lr0p001_ema0p999` |
| pipeline 原始 generation/eval | `autodl_5090_2/results/pipeline/{model}_{dataset}_{method}_alpha...json` 与 `autodl_5090_2/results/pipeline/eval_{model}_{dataset}_{method}_alpha...json` | 每个 generation 文件对应一个 eval 文件；最终汇总由 Step12/collect 生成 |
| pipeline 最终 EM/F1 总表 | `autodl_5090_2/results/pipeline/pipeline_eval_summary.csv` | 最终 source-of-truth；共 84 行，Gemma2-2B 42 行、Qwen3-4B 42 行 |
| pipeline tau 对比表 | `results/analysis/pipeline_plugin_comparison_by_tau.csv` / `.md` | 从 `autodl_5090_2/results/pipeline/pipeline_eval_summary.csv` 派生；baseline 固定取对应非 gated 方法，plugin 按 tau=`0.5/0.4/0.3` 展开 |
| pipeline best-tau summary | 本计划 0.4b；派生自 `autodl_5090_2/results/pipeline/pipeline_eval_summary.csv` | 对每个 `(model, dataset, baseline/plugin)` 按 `F1 plugin` 选择最佳 tau |
| 当前最强 baseline 摘要 | 本计划后部摘要；派生自 `autodl_5090_2/results/pipeline/pipeline_eval_summary.csv` | 只在非 `csv_gated_*` 方法中按 F1 取最佳；当前最终表不包含 Gemma2-9B pipeline |
| Logit Lens 主表 | `results/logit_lens/{gemma2b,gemma9b,qwen3_4b}_logit_lens.json` | final P 由最后一层 `mean_logP_combined` 取 `exp`；peak layer/value 取 JSON 中的 `delta_gold_peak_*` |
| retrieved-doc Logit Lens 诊断 | `results/logit_lens_retrieved/gemma2b_retrieved_logit_lens_nq.json` | 当前只作为诊断证据；Gemma2B/NQ retrieved relevant peak layer 来自 `retrieved_relevant_peak_layer` |
| 原始表征重叠图 | `data/hidden_states_retrieved/{gemma2b,qwen3_4b}/eval_hidden_states.pt`，脚本 `scripts/plot_original_representation_overlap_retrieved.py`，输出 `results/case_analysis/original_representation_overlap/` | 只取原始 cached hidden states；Gemma2B 取 `cls_layer=18`，Qwen3-4B 取 `cls_layer=22`；每个 dataset/class 抽 160 条平衡样本；不包含 CSV/after-method 表征 |

### 0.2 旧 gold-vs-distractor：CSV 与基线结果

数据：`data/final/train.json` / `data/final/eval.json`。指标为 eval AUROC / Accuracy。该口径用于证明 hidden-state separability，不作为 Step11 gate 输入。

| Model | Classifier | Best | combined AUROC | NQ AUROC | TriviaQA AUROC | combined Acc | NQ Acc | TriviaQA Acc |
|---|---|---|---|---|---|---|---|---|
| External | DPR dot | - | 0.2919 | 0.4022 | 0.1748 | 0.5000 | 0.5000 | 0.5000 |
| External | CRAG | - | 0.4519 | 0.4393 | 0.4631 | 0.5000 | 0.5000 | 0.5000 |
| External | Self-RAG | - | 0.6451 | 0.6435 | 0.6527 | 0.6046 | 0.5972 | 0.6104 |
| External | Cross-Encoder | MiniLM-L6 | 0.9922 | 0.9938 | 0.9906 | 0.9524 | 0.9630 | 0.9442 |
| Gemma2-2B | LR | layer_5 | 0.8029 | 0.7703 | 0.7531 | 0.7446 | 0.7114 | 0.7039 |
| Gemma2-2B | MLP | layer_5 | 0.8568 | 0.8124 | 0.8401 | 0.7779 | 0.7407 | 0.7743 |
| Gemma2-2B | Centroid | layer_5 | 0.7458 | 0.7350 | 0.7175 | 0.6855 | 0.6651 | 0.6638 |
| Gemma2-2B | Mass-Mean | layer_5 | 0.7456 | 0.7353 | 0.7168 | 0.6875 | 0.6590 | 0.6650 |
| Gemma2-2B | Low-rank Rep. Head | final/r8/a16 | 0.8891 | 0.9036 | 0.8785 | 0.8152 | 0.8210 | 0.8107 |
| Gemma2-2B | Ours / CSV | inject_1_cls_16 | 0.9964 | 0.9970 | 0.9957 | 0.9592 | 0.9614 | 0.9575 |
| Gemma2-9B | LR | layer_29 | 0.7605 | 0.6849 | 0.8241 | 0.7052 | 0.6343 | 0.7524 |
| Gemma2-9B | MLP | layer_31 | 0.7702 | 0.6723 | 0.8089 | 0.7052 | 0.6250 | 0.7415 |
| Gemma2-9B | Centroid | layer_29 | 0.7415 | 0.6515 | 0.7964 | 0.6712 | 0.5988 | 0.7148 |
| Gemma2-9B | Mass-Mean | layer_29 | 0.7404 | 0.6513 | 0.7961 | 0.6698 | 0.5957 | 0.7160 |
| Gemma2-9B | Ours / CSV | inject_0_cls_30 | 0.9974 | 0.9983 | 0.9962 | 0.9654 | 0.9722 | 0.9600 |
| Qwen3-4B | LR | layer_23 | 0.8127 | 0.7729 | 0.8391 | 0.7486 | 0.6806 | 0.7779 |
| Qwen3-4B | MLP | layer_23 | 0.8325 | 0.7357 | 0.8483 | 0.7636 | 0.6698 | 0.7803 |
| Qwen3-4B | Centroid | layer_23 | 0.7768 | 0.7527 | 0.8126 | 0.7154 | 0.6883 | 0.7439 |
| Qwen3-4B | Mass-Mean | layer_23 | 0.7777 | 0.7533 | 0.8125 | 0.7154 | 0.6883 | 0.7427 |
| Qwen3-4B | Low-rank Rep. Head | final/r8/a16 | 0.9309 | 0.9455 | 0.9209 | 0.8655 | 0.8796 | 0.8544 |
| Qwen3-4B | Ours / CSV | inject_6_cls_24 | 0.9961 | 0.9966 | 0.9956 | 0.9552 | 0.9583 | 0.9527 |

注意：本表的 DPR baseline 是旧 gold-vs-distractor 口径下重新计算的 DPR encoder dot，相比 labeled gold evidence vs retrieved distractor 呈反相关；不要与 retrieved-doc 表中的 `doc.score` / retriever-score baseline 混用。Cross-Encoder 是单独训练的外部监督 reranker/classifier，它直接编码 question-document token interaction；在 gold-vs-distractor 口径下正类是人工 gold evidence，负类是 distracting passage，因此该任务非常适合 cross-encoder，分数接近 CSV 是合理的，但它不反映目标 LLM 内部 hidden states 是否自然可分。

结论：旧 CSV 在三个模型上都达到约 0.996 AUROC，显著强于 LR/MLP/Centroid/Mass-Mean、Low-rank Rep. Head 与多数外部 evaluator baseline；Cross-Encoder 在 gold-vs-distractor 上也很强，因为它是直接监督训练的 pairwise reranker。该表中 CSV 的核心作用仍是证明目标 LLM hidden-state separability / CSV amplification，而不是证明比所有外部 reranker 都绝对更强。

### 0.3 新 retrieved-doc：CSV_R 与基线结果

数据：`data/final_retrieved/train.json` / `data/final_retrieved/eval.json`。指标为 eval AUROC / Accuracy。该口径是 Step11 gate 主线，用于判断真实 pipeline 里的 top-1 retrieved doc 是否 relevant。

| Model | Classifier | Best | combined AUROC | NQ AUROC | TriviaQA AUROC | combined Acc | NQ Acc | TriviaQA Acc |
|---|---|---|---|---|---|---|---|---|
| External | DPR retriever score | - | 0.6376 | 0.6042 | 0.6223 | 0.6253 | 0.6528 | 0.5931 |
| External | CRAG | - | 0.5171 | 0.4736 | 0.5759 | 0.4244 | 0.3385 | 0.5249 |
| External | Self-RAG | - | 0.7688 | 0.7669 | 0.8263 | 0.6984 | 0.6615 | 0.7416 |
| External | Cross-Encoder | MiniLM-L6 | 0.7985 | 0.7744 | 0.8246 | 0.7504 | 0.7405 | 0.7620 |
| Gemma2-2B | LR | layer_18 | 0.8497 | 0.7996 | 0.8941 | 0.7536 | 0.7153 | 0.8199 |
| Gemma2-2B | MLP | layer_18 | 0.8486 | 0.7949 | 0.8925 | 0.7607 | 0.7005 | 0.8220 |
| Gemma2-2B | Centroid | layer_18 | 0.8403 | 0.7638 | 0.8779 | 0.7696 | 0.6502 | 0.7955 |
| Gemma2-2B | Mass-Mean | layer_18 | 0.8403 | 0.7637 | 0.8781 | 0.7756 | 0.6545 | 0.7976 |
| Gemma2-2B | Low-rank Rep. Head | final/r8/a16 | 0.8284 | 0.7817 | 0.8609 | 0.7494 | 0.7222 | 0.7813 |
| Gemma2-2B | Ours / CSV_R | inject_14_cls_18 | 0.9029 | 0.8558 | 0.9379 | 0.8375 | 0.8116 | 0.8678 |
| Gemma2-9B | LR | layer_27 | 0.8893 | 0.8396 | 0.9321 | 0.8005 | 0.7587 | 0.8616 |
| Gemma2-9B | MLP | layer_29 | 0.8995 | 0.8379 | 0.9262 | 0.8108 | 0.7587 | 0.8576 |
| Gemma2-9B | Centroid | layer_27 | 0.8780 | 0.8238 | 0.9237 | 0.8108 | 0.7344 | 0.8566 |
| Gemma2-9B | Mass-Mean | layer_27 | 0.8781 | 0.8239 | 0.9238 | 0.8173 | 0.7457 | 0.8576 |
| Qwen3-4B | LR | layer_23 | 0.8918 | 0.8587 | 0.9272 | 0.8262 | 0.7951 | 0.8678 |
| Qwen3-4B | MLP | layer_23 | 0.8950 | 0.8580 | 0.9180 | 0.8351 | 0.7899 | 0.8545 |
| Qwen3-4B | Centroid | layer_23 | 0.8857 | 0.8401 | 0.9161 | 0.8201 | 0.7726 | 0.8423 |
| Qwen3-4B | Mass-Mean | layer_23 | 0.8860 | 0.8402 | 0.9162 | 0.8220 | 0.7752 | 0.8423 |
| Qwen3-4B | Low-rank Rep. Head | final/r8/a16 | 0.8810 | 0.8335 | 0.9117 | 0.8169 | 0.7830 | 0.8566 |
| Qwen3-4B | Ours / CSV_R | inject_21_cls_22 | 0.9159 | 0.8821 | 0.9409 | 0.8553 | 0.8403 | 0.8728 |

结果文件：

```text
results/probe_retrieved/*.json
results/probe_retrieved/dpr_baseline.json  # score_source=retriever_score, 使用 data/final_retrieved/* 中的 doc.score
results/cross_encoder_retrieved/*/results.json
results/lowrank_rep_head_retrieved/*/results.json
results/pipeline/gemma2b_retrieved_csv_eval_validation.json
results/pipeline/qwen3_4b_retrieved_csv_eval_validation.json
results/csv_retrieved_qwen3_4b_best_for_pipeline/
```

注意：本表的 DPR baseline 是 retrieved-doc 口径下的 `doc.score` / retriever-score，不重新计算 encoder dot。它直接对应 DPR/FAISS top-1 检索时保存的分数。

结论：retrieved-doc 口径更贴近真实 RAG pipeline，也更难；CSV_R 在 Gemma2-2B 和 Qwen3-4B 上都稳定强于同模型 LR / MLP / Centroid / Mass-Mean / Low-rank Rep. Head，也强于 DPR retriever-score、CRAG、Self-RAG evaluator 与 Cross-Encoder。Gemma2-9B 的 retrieved-doc probe baselines 已补齐，但 CSV_R 暂未作为主线结果写入本表。

#### 0.3a 新增强基线：Cross-Encoder 与 Low-rank Representation Head

本节新增两类更强/更接近 reviewer 建议的 utility classification baseline。统一汇总文件：

```text
results/analysis/additional_baselines_summary.md
results/analysis/additional_baselines_summary.csv
```

推荐表名与引用：

| 表中名称 | 适用行 | 引用建议 | 备注 |
|---|---|---|---|
| Cross-Encoder | `answer_mode=none` | Nogueira & Cho (2019); Wang et al. (2020) | 标准 BERT-style passage reranking / cross-encoder relevance classifier |
| Low-rank Rep. Head | final hidden state low-rank head | Hu et al. (2021); Wu et al. (2024) | 受 LoRA/ReFT 启发的 low-rank representation-head baseline，但不是 full LoRA / full ReFT |

引用条目：

- Nogueira and Cho, 2019, *Passage Re-ranking with BERT*: https://arxiv.org/abs/1901.04085
- Wang et al., 2020, *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*: https://arxiv.org/abs/2002.10957
- Hu et al., 2021, *LoRA: Low-Rank Adaptation of Large Language Models*: https://arxiv.org/abs/2106.09685
- Wu et al., 2024, *ReFT: Representation Finetuning for Language Models*: https://arxiv.org/abs/2404.03592

**Cross-Encoder answer-support classifier**

脚本：`scripts/step15a_train_answer_support_cross_encoder.py`。

模型为 `cross-encoder/ms-marco-MiniLM-L6-v2`，训练 3 epochs，batch size 16，learning rate 2e-5；每个样本输入为 `(question, document)` 的 cross-encoder pair，输出 `P(relevant)`，阈值 0.5 算 accuracy。

脚本支持两个数据设置：

| Setting | 输入数据 | 代码构造方式 | 输出目录 |
|---|---|---|---|
| `--setting gold` | `data/final/{train,eval}.json` | 每条原始样本展开成两条：`relevant_doc -> label 1`，`distracting_doc -> label 0` | `results/cross_encoder_gold/` |
| `--setting retrieved` | `data/final_retrieved/{train,eval}.json` | 每条样本直接读取 `doc` 和 `label` | `results/cross_encoder_retrieved/` |

正式台账只保留 `--answer_mode none`：输入 A 只包含 question，输入 B 为 document。磁盘上曾生成过答案感知诊断结果，但不纳入 PLAN 或论文表格，因为当前只需要两个公平数据设置。

Cross-Encoder retrieved-doc 结果：

| Method | Combined AUROC | Combined Acc | NQ AUROC | NQ Acc | TriviaQA AUROC | TriviaQA Acc | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Cross-Encoder | 0.7985 | 0.7504 | 0.7744 | 0.7405 | 0.8246 | 0.7620 | `results/cross_encoder_retrieved/cross-encoder_ms-marco-MiniLM-L6-v2_none/results.json` |

Cross-Encoder gold-vs-distractor 结果：

| Method | Combined AUROC | Combined Acc | NQ AUROC | NQ Acc | TriviaQA AUROC | TriviaQA Acc | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Cross-Encoder | 0.9922 | 0.9524 | 0.9938 | 0.9630 | 0.9906 | 0.9442 | `results/cross_encoder_gold/cross-encoder_ms-marco-MiniLM-L6-v2_gold_none/results.json` |

解释：gold-vs-distractor 的正类是人工 gold evidence，负类是 distracting passage；Cross-Encoder 直接进行 question-document token-level interaction，并且在该 train split 上监督训练，因此该 setting 下分数很高是预期现象。retrieved-doc setting 更接近真实 pipeline，样本分布更难，因此同一 Cross-Encoder 在 retrieved-doc 上明显下降。

**Low-rank Representation Head**

脚本：`scripts/step15b_train_lowrank_rep_head.py`。

该 baseline 冻结 LLM，不重新 forward/backward 整个模型；它读取已提取的 hidden states，并在 final hidden representation 上训练一个小的低秩 residual adapter + classifier：

```text
z = h + (alpha / rank) * B(Ah)
logits = Wz + b
```

主设置：`--layer -1`，`rank=8`，`alpha=16`，batch size 256，learning rate 1e-3，max epochs 30，early stopping patience 5。它是 parameter-efficient representation adaptation 风格的 **Low-rank Representation Head**，不要写成 full LoRA 或 full ReFT；因为它没有把 LoRA 模块插进 LLM 权重，也没有在 LLM forward 过程中做 LoReFT intervention。

输入数据：

| Setting | Hidden-state source | 说明 |
|---|---|---|
| gold-vs-distractor | `data/hidden_states/{model}/{train,eval}_hidden_states.pt` | 旧 gold document vs distracting document 口径 |
| retrieved-doc | `data/hidden_states_retrieved/{model}/{train,eval}_hidden_states.pt` | pipeline-matched top-1 retrieved doc 口径 |

Low-rank Representation Head 结果：

| Setting | Model | Combined AUROC | Combined Acc | NQ AUROC | NQ Acc | TriviaQA AUROC | TriviaQA Acc | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Gold-vs-distractor | Gemma2-2B | 0.8891 | 0.8152 | 0.9036 | 0.8210 | 0.8785 | 0.8107 | `results/lowrank_rep_head_gold/gemma2b_gold_final_rank8_alpha16p0/results.json` |
| Gold-vs-distractor | Qwen3-4B | 0.9309 | 0.8655 | 0.9455 | 0.8796 | 0.9209 | 0.8544 | `results/lowrank_rep_head_gold/qwen3_4b_gold_final_rank8_alpha16p0/results.json` |
| Retrieved-doc | Gemma2-2B | 0.8284 | 0.7494 | 0.7817 | 0.7222 | 0.8609 | 0.7813 | `results/lowrank_rep_head_retrieved/gemma2b_retrieved_final_rank8_alpha16p0/results.json` |
| Retrieved-doc | Qwen3-4B | 0.8810 | 0.8169 | 0.8335 | 0.7830 | 0.9117 | 0.8566 | `results/lowrank_rep_head_retrieved/qwen3_4b_retrieved_final_rank8_alpha16p0/results.json` |

解释口径：

- Cross-Encoder 是一个不使用 LLM hidden states 的强外部 pairwise relevance classifier；正式台账只比较 gold-vs-distractor 与 retrieved-doc 两个公平数据设置。
- Low-rank Representation Head 测试“只在 frozen representation 顶部加更强小参数 head”是否能替代 CSV；当前结果低于对应 CSV_R/Ours，支持 CSV 的 steering/prototype 机制不是普通 low-rank classifier head 能完全解释的。

#### 0.3b Retrieved-doc 2×2 ablation：CSV × Prototype

数据：`data/final_retrieved/train.json` / `data/final_retrieved/eval.json`。指标为 Combined eval AUROC / Accuracy。该消融固定为 retrieved-doc gate 口径，只比较分类器内部设计，不涉及 Step11 下游 QA EM/F1。

消融数据与结果路径：

```text
主仓库原始分类数据:
  train: data/final_retrieved/train.json
  eval:  data/final_retrieved/eval.json

迁移包副本（用于补跑 (c) CSV + LR head）:
  train: c_ablation_retrieved_lr_20260513/data/final_retrieved/train.json
  eval:  c_ablation_retrieved_lr_20260513/data/final_retrieved/eval.json

(a) / (b) frozen hidden states:
  data/hidden_states_retrieved/gemma2b/train_hidden_states.pt
  data/hidden_states_retrieved/gemma2b/eval_hidden_states.pt
  data/hidden_states_retrieved/qwen3_4b/train_hidden_states.pt
  data/hidden_states_retrieved/qwen3_4b/eval_hidden_states.pt

(c) CSV + LR head 结果:
  c_ablation_retrieved_lr_20260513/results/ablation_csv_nonprototype/gemma2b_retrieved_csv_nonprototype_lr.json
  c_ablation_retrieved_lr_20260513/results/ablation_csv_nonprototype/qwen3_4b_retrieved_csv_nonprototype_lr.json

Ours / full method 结果:
  results/pipeline/gemma2b_retrieved_csv_eval_validation.json
  results/pipeline/qwen3_4b_retrieved_csv_eval_validation.json
```

| Index | CSV | Prototype | Gemma2-2B AUROC | Gemma2-2B Acc. | Qwen3-4B AUROC | Qwen3-4B Acc. |
|---|---|---|---:|---:|---:|---:|
| (a) ordinary probe | ✗ | ✗ | 0.8497 | 0.7536 | 0.8918 | 0.8262 |
| (b) prototype classifier without CSV | ✗ | ✓ | 0.8403 | 0.7696 | 0.8857 | 0.8201 |
| (c) CSV with non-prototype head | ✓ | ✗ | 0.9000 | 0.8276 | 0.9103 | 0.8407 |
| Ours / full method | ✓ | ✓ | 0.9029 | 0.8375 | 0.9159 | 0.8553 |

填表口径：

```text
(a) ordinary probe:
  results/probe_retrieved/{model}_probe_results_lr.json
  按 combined AUROC 选 best layer；Gemma2B layer_18，Qwen3-4B layer_23

(b) prototype classifier without CSV:
  results/probe_retrieved/{model}_probe_results_centroid.json
  按 combined AUROC 选 best layer；Gemma2B layer_18，Qwen3-4B layer_23

(c) CSV with non-prototype head:
  c_ablation_retrieved_lr_20260513/results/ablation_csv_nonprototype/{model}_retrieved_csv_nonprototype_lr.json
  使用同一 CSV_R best checkpoint 注入 CSV 后训练 LR head；
  Gemma2B CSV_R=inject_14_cls_18，Qwen3-4B CSV_R=inject_21_cls_22

Ours / full method:
  results/pipeline/{model}_retrieved_csv_eval_validation.json
  使用 CSV_R checkpoint 中的 prototype / centroid decision rule
```

结论：CSV 注入本身带来主要增益，`CSV + LR` 已明显强于无 CSV 的 LR / Centroid；full method 进一步提升 accuracy，说明 CSV-modified representation 与 prototype decision rule 的组合仍有额外收益。

### 0.3c 人工标签校验：utility label quality

目的：由于 utility label 由自动流程构造/标注，需要人工抽样检查 label quality。人工判断标准为：给定 question、reference answer 与 candidate document，判断该 document 是否包含足够证据来支持或直接推断答案。若 `manual_label == auto_label`，则记为 agreement。

自动标签含义：

```text
auto_label=1 / relevant: document is answer-supporting
auto_label=0 / distracting: document is not answer-supporting
```

抽样口径：

| Setting | 抽样方式 | 样本数 | 原因 |
|---|---|---:|---|
| Gold-vs-distractor | 每个数据集抽 50 条 distractor label；NQ=50，TriviaQA=50 | 100 | gold passage 是 answer-supporting by construction，主要噪声风险是 distractor 里意外包含答案证据 |
| Retrieved-document | 每个数据集抽 25 条 support + 25 条 distractor；NQ=50，TriviaQA=50 | 100 | retrieved-doc 的 support/distractor 都来自自动标注，需要平衡检查两类标签 |

文件位置：

```text
抽样总表: results/manual_validation/manual_validation_samples.csv
抽样 JSONL: results/manual_validation/manual_validation_samples.jsonl
Gold-dist split: results/manual_validation/manual_validation_gold_vs_distractor.csv
Retrieved split: results/manual_validation/manual_validation_retrieved_document.csv
简化标注表: results/manual_validation/manual_validation_annotation_sheet.csv
网页标注工具: results/manual_validation/annotation_site/
人工完成文件: results/manual_validation/manual_validation_completed.csv
Agreement 汇总: results/manual_validation/manual_validation_summary.csv
标注说明: results/manual_validation/ANNOTATION_GUIDE.md
```

人工校验主表：

| Setting | Sample | # | Agreement (%) |
|---|---|---:|---:|
| Gold-dist. | Distractor | 100 | 95.0 |
| Retrieved | Support | 50 | 74.0 |
| Retrieved | Distractor | 50 | 92.0 |
| Retrieved | Overall | 100 | 83.0 |

细分 sanity check：

| Setting | Dataset | Sample | # | Correct | Agreement (%) |
|---|---|---|---:|---:|---:|
| Gold-dist. | NQ | Distractor | 50 | 46 | 92.0 |
| Gold-dist. | TriviaQA | Distractor | 50 | 49 | 98.0 |
| Retrieved | NQ | Support | 25 | 20 | 80.0 |
| Retrieved | NQ | Distractor | 25 | 23 | 92.0 |
| Retrieved | TriviaQA | Support | 25 | 17 | 68.0 |
| Retrieved | TriviaQA | Distractor | 25 | 23 | 92.0 |

推荐论文表格：

```latex
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.05}
\caption{\textbf{Manual validation of utility labels.} Gold-vs-distractor validation checks only distractor labels; retrieved-document validation uses balanced samples of answer-supporting and distracting documents.}
\begin{tabular}{llcc}
\toprule
\textbf{Setting} & \textbf{Sample} & \textbf{\#} & \textbf{Agr. (\%)} \\
\midrule
Gold-dist. & Distractor & 100 & 95.0 \\
Retrieved & Support & 50 & 74.0 \\
Retrieved & Distractor & 50 & 92.0 \\
Retrieved & Overall & 100 & 83.0 \\
\bottomrule
\end{tabular}
\label{tab:label_validation}
\end{table}
```

写作口径：Gold-vs-distractor 的 distractor label agreement 较高，说明 constructed negative passage 通常确实不支持答案；retrieved-doc overall agreement 为 83.0%，说明真实检索口径存在更明显 label noise，尤其 retrieved support 标签更难，主要由 TriviaQA support agreement 较低拉低。这可以作为 retrieved-doc setting 更 noisy、更贴近真实 pipeline 的补充证据。

#### 0.3d 原始表征重叠图（clean / before CSV only）

目的：补充一组 qualitative visualization，展示原始 retrieved-doc hidden representations 中 relevant 与 distracting documents 并未清晰分开。该图只画原始表征，不画使用本方法后的表征，避免与 CSV intervention 结果混在同一图里。

脚本与输出：

```text
脚本: scripts/plot_original_representation_overlap_retrieved.py
输入: data/hidden_states_retrieved/{gemma2b,qwen3_4b}/eval_hidden_states.pt
输出: results/case_analysis/original_representation_overlap/
```

四张主图：

```text
results/case_analysis/original_representation_overlap/gemma2b_nq_original_representation_overlap.png
results/case_analysis/original_representation_overlap/gemma2b_triviaqa_original_representation_overlap.png
results/case_analysis/original_representation_overlap/qwen3_4b_nq_original_representation_overlap.png
results/case_analysis/original_representation_overlap/qwen3_4b_triviaqa_original_representation_overlap.png
```

同目录下已同时导出 SVG/PDF，用于论文排版。配色采用粉色 distracting、蓝色 answer-supporting/relevant，点与协方差椭圆均为半透明，背景为浅粉色；图中不显示坐标轴、legend 或方法后对比 panel。

画图口径：

| Model | Dataset | Split | cls layer | cached hidden index | Samples | Distracting | Relevant |
|---|---|---|---:|---:|---:|---:|---:|
| Gemma2-2B | NQ | eval | 18 | 19 | 320 | 160 | 160 |
| Gemma2-2B | TriviaQA | eval | 18 | 19 | 320 | 160 | 160 |
| Qwen3-4B | NQ | eval | 22 | 23 | 320 | 160 | 160 |
| Qwen3-4B | TriviaQA | eval | 22 | 23 | 320 | 160 | 160 |

PCA 2D separability diagnostic（仅用于说明图中原始表征重叠，不作为主实验指标）：

| Model | Dataset | Silhouette | Separation ratio |
|---|---|---:|---:|
| Gemma2-2B | NQ | 0.029 | 0.370 |
| Gemma2-2B | TriviaQA | 0.131 | 0.916 |
| Qwen3-4B | NQ | 0.060 | 0.539 |
| Qwen3-4B | TriviaQA | 0.166 | 0.973 |

指标文件：`results/case_analysis/original_representation_overlap/original_representation_overlap_metrics.csv`。

写作口径：These clean hidden-state projections show substantial overlap between answer-supporting and distracting retrieved documents before intervention, motivating the need for CSV-based representation steering/gating. Do not describe this figure as the post-CSV representation; it is explicitly before-method / original representation only.

### 0.4 Pipeline plugin：Gemma2-2B 与 Qwen3-4B 结果

数据：`data/final/test_retrieval_nq.json` 与 `data/final/test_retrieval_triviaqa.json`。指标为 EM / F1。Baseline 固定口径：CAD `alpha=0.5`；ACD `alpha=1.0`；greedy decoding；`max_new_tokens=20`；`max_input_length=512`。Plugin 使用各模型 CSV_R best，`alpha=0.5`，tau sweep=`0.5/0.4/0.3`。

结果来源：

```text
最终 summary: autodl_5090_2/results/pipeline/pipeline_eval_summary.csv  # 84 行总表；Gemma2-2B 42 行，Qwen3-4B 42 行
Gemma2-2B 本地镜像: results/pipeline/pipeline_eval_summary.csv  # 42 行，仅用于交叉检查
完整 tau 表: results/analysis/pipeline_plugin_comparison_by_tau.md  # 由最终 summary 派生
完整 tau CSV: results/analysis/pipeline_plugin_comparison_by_tau.csv  # 由最终 summary 派生
Qwen 单模型表: results/analysis/qwen3_4b_pipeline_plugin_comparison.md
```

当前 `autodl_5090_2/results/pipeline/pipeline_eval_summary.csv` 是 84 行总表，包含 Gemma2-2B 42 行与 Qwen3-4B 42 行；其中 Qwen3-4B `triviaqa / dola` 的 eval 是从已存在 generation 文件补评得到。`autodl_5090_2` 覆盖 `autodl_5090_1` 的所有 pipeline/probe/probe_retrieved/csv_retrieved 文件，并额外包含 Qwen pipeline。

#### 0.4a 完整 tau sweep 表

#### tau=0.5

Gemma2-2B

| dataset | Baseline | Plugin | EM baseline | EM plugin | Delta EM | F1 baseline | F1 plugin | Delta F1 |
|---|---|---|---|---|---|---|---|---|
| NQ | vanilla_rag | csv_gated_vanilla_rag | 0.2576 | 0.2593 | +0.0017 | 0.3647 | 0.3625 | -0.0022 |
| NQ | cad_fixed | csv_gated_cad | 0.1701 | 0.2072 | +0.0371 | 0.2403 | 0.2915 | +0.0512 |
| NQ | acd | csv_gated_acd | 0.2499 | 0.2019 | -0.0479 | 0.3592 | 0.2977 | -0.0614 |
| NQ | context_ucd | csv_gated_context_ucd | 0.1133 | 0.1687 | +0.0554 | 0.1694 | 0.2441 | +0.0747 |
| NQ | dola | csv_gated_dola | 0.0562 | 0.1114 | +0.0551 | 0.1440 | 0.2073 | +0.0633 |
| TriviaQA | vanilla_rag | csv_gated_vanilla_rag | 0.4329 | 0.4926 | +0.0598 | 0.5183 | 0.5692 | +0.0509 |
| TriviaQA | cad_fixed | csv_gated_cad | 0.3058 | 0.4217 | +0.1159 | 0.3899 | 0.5038 | +0.1139 |
| TriviaQA | acd | csv_gated_acd | 0.4991 | 0.4947 | -0.0043 | 0.5760 | 0.5649 | -0.0111 |
| TriviaQA | context_ucd | csv_gated_context_ucd | 0.2029 | 0.3601 | +0.1573 | 0.2769 | 0.4405 | +0.1636 |
| TriviaQA | dola | csv_gated_dola | 0.1885 | 0.3425 | +0.1541 | 0.2890 | 0.4368 | +0.1478 |

Qwen3-4B

| dataset | Baseline | Plugin | EM baseline | EM plugin | Delta EM | F1 baseline | F1 plugin | Delta F1 |
|---|---|---|---|---|---|---|---|---|
| NQ | vanilla_rag | csv_gated_vanilla_rag | 0.2709 | 0.2803 | +0.0094 | 0.3636 | 0.3718 | +0.0082 |
| NQ | cad_fixed | csv_gated_cad | 0.2136 | 0.2454 | +0.0319 | 0.3082 | 0.3425 | +0.0343 |
| NQ | acd | csv_gated_acd | 0.2806 | 0.2305 | -0.0501 | 0.3712 | 0.3199 | -0.0513 |
| NQ | context_ucd | csv_gated_context_ucd | 0.1460 | 0.1945 | +0.0485 | 0.2358 | 0.2945 | +0.0586 |
| NQ | dola | csv_gated_dola | 0.0078 | 0.0753 | +0.0676 | 0.0609 | 0.1505 | +0.0896 |
| TriviaQA | vanilla_rag | csv_gated_vanilla_rag | 0.3750 | 0.3731 | -0.0019 | 0.4667 | 0.4642 | -0.0025 |
| TriviaQA | cad_fixed | csv_gated_cad | 0.3195 | 0.3412 | +0.0217 | 0.4108 | 0.4433 | +0.0325 |
| TriviaQA | acd | csv_gated_acd | 0.3523 | 0.2957 | -0.0567 | 0.4494 | 0.3935 | -0.0559 |
| TriviaQA | context_ucd | csv_gated_context_ucd | 0.2494 | 0.3013 | +0.0520 | 0.3415 | 0.4110 | +0.0695 |
| TriviaQA | dola | csv_gated_dola | 0.0279 | 0.1363 | +0.1084 | 0.0931 | 0.2343 | +0.1412 |

#### tau=0.4

Gemma2-2B

| dataset | Baseline | Plugin | EM baseline | EM plugin | Delta EM | F1 baseline | F1 plugin | Delta F1 |
|---|---|---|---|---|---|---|---|---|
| NQ | vanilla_rag | csv_gated_vanilla_rag | 0.2576 | 0.2593 | +0.0017 | 0.3647 | 0.3642 | -0.0004 |
| NQ | cad_fixed | csv_gated_cad | 0.1701 | 0.2044 | +0.0343 | 0.2403 | 0.2875 | +0.0472 |
| NQ | acd | csv_gated_acd | 0.2499 | 0.2022 | -0.0476 | 0.3592 | 0.2984 | -0.0608 |
| NQ | context_ucd | csv_gated_context_ucd | 0.1133 | 0.1637 | +0.0504 | 0.1694 | 0.2368 | +0.0673 |
| NQ | dola | csv_gated_dola | 0.0562 | 0.1066 | +0.0504 | 0.1440 | 0.2021 | +0.0581 |
| TriviaQA | vanilla_rag | csv_gated_vanilla_rag | 0.4329 | 0.4916 | +0.0587 | 0.5183 | 0.5690 | +0.0507 |
| TriviaQA | cad_fixed | csv_gated_cad | 0.3058 | 0.4181 | +0.1123 | 0.3899 | 0.5003 | +0.1104 |
| TriviaQA | acd | csv_gated_acd | 0.4991 | 0.4945 | -0.0046 | 0.5760 | 0.5653 | -0.0108 |
| TriviaQA | context_ucd | csv_gated_context_ucd | 0.2029 | 0.3542 | +0.1513 | 0.2769 | 0.4343 | +0.1573 |
| TriviaQA | dola | csv_gated_dola | 0.1885 | 0.3373 | +0.1489 | 0.2890 | 0.4322 | +0.1432 |

Qwen3-4B

| dataset | Baseline | Plugin | EM baseline | EM plugin | Delta EM | F1 baseline | F1 plugin | Delta F1 |
|---|---|---|---|---|---|---|---|---|
| NQ | vanilla_rag | csv_gated_vanilla_rag | 0.2709 | 0.2831 | +0.0122 | 0.3636 | 0.3759 | +0.0123 |
| NQ | cad_fixed | csv_gated_cad | 0.2136 | 0.2443 | +0.0307 | 0.3082 | 0.3435 | +0.0353 |
| NQ | acd | csv_gated_acd | 0.2806 | 0.2332 | -0.0474 | 0.3712 | 0.3230 | -0.0482 |
| NQ | context_ucd | csv_gated_context_ucd | 0.1460 | 0.1906 | +0.0446 | 0.2358 | 0.2915 | +0.0557 |
| NQ | dola | csv_gated_dola | 0.0078 | 0.0690 | +0.0612 | 0.0609 | 0.1433 | +0.0824 |
| TriviaQA | vanilla_rag | csv_gated_vanilla_rag | 0.3750 | 0.3741 | -0.0009 | 0.4667 | 0.4654 | -0.0013 |
| TriviaQA | cad_fixed | csv_gated_cad | 0.3195 | 0.3412 | +0.0217 | 0.4108 | 0.4435 | +0.0327 |
| TriviaQA | acd | csv_gated_acd | 0.3523 | 0.2959 | -0.0564 | 0.4494 | 0.3938 | -0.0556 |
| TriviaQA | context_ucd | csv_gated_context_ucd | 0.2494 | 0.2996 | +0.0502 | 0.3415 | 0.4095 | +0.0681 |
| TriviaQA | dola | csv_gated_dola | 0.0279 | 0.1302 | +0.1023 | 0.0931 | 0.2279 | +0.1348 |

#### tau=0.3

Gemma2-2B

| dataset | Baseline | Plugin | EM baseline | EM plugin | Delta EM | F1 baseline | F1 plugin | Delta F1 |
|---|---|---|---|---|---|---|---|---|
| NQ | vanilla_rag | csv_gated_vanilla_rag | 0.2576 | 0.2598 | +0.0022 | 0.3647 | 0.3652 | +0.0006 |
| NQ | cad_fixed | csv_gated_cad | 0.1701 | 0.2019 | +0.0319 | 0.2403 | 0.2828 | +0.0425 |
| NQ | acd | csv_gated_acd | 0.2499 | 0.2033 | -0.0465 | 0.3592 | 0.3001 | -0.0591 |
| NQ | context_ucd | csv_gated_context_ucd | 0.1133 | 0.1609 | +0.0476 | 0.1694 | 0.2312 | +0.0618 |
| NQ | dola | csv_gated_dola | 0.0562 | 0.1011 | +0.0449 | 0.1440 | 0.1951 | +0.0511 |
| TriviaQA | vanilla_rag | csv_gated_vanilla_rag | 0.4329 | 0.4892 | +0.0563 | 0.5183 | 0.5674 | +0.0491 |
| TriviaQA | cad_fixed | csv_gated_cad | 0.3058 | 0.4127 | +0.1069 | 0.3899 | 0.4952 | +0.1053 |
| TriviaQA | acd | csv_gated_acd | 0.4991 | 0.4949 | -0.0042 | 0.5760 | 0.5659 | -0.0102 |
| TriviaQA | context_ucd | csv_gated_context_ucd | 0.2029 | 0.3463 | +0.1435 | 0.2769 | 0.4263 | +0.1494 |
| TriviaQA | dola | csv_gated_dola | 0.1885 | 0.3307 | +0.1422 | 0.2890 | 0.4255 | +0.1365 |

Qwen3-4B

| dataset | Baseline | Plugin | EM baseline | EM plugin | Delta EM | F1 baseline | F1 plugin | Delta F1 |
|---|---|---|---|---|---|---|---|---|
| NQ | vanilla_rag | csv_gated_vanilla_rag | 0.2709 | 0.2848 | +0.0139 | 0.3636 | 0.3780 | +0.0144 |
| NQ | cad_fixed | csv_gated_cad | 0.2136 | 0.2446 | +0.0310 | 0.3082 | 0.3444 | +0.0361 |
| NQ | acd | csv_gated_acd | 0.2806 | 0.2349 | -0.0457 | 0.3712 | 0.3249 | -0.0463 |
| NQ | context_ucd | csv_gated_context_ucd | 0.1460 | 0.1889 | +0.0429 | 0.2358 | 0.2900 | +0.0541 |
| NQ | dola | csv_gated_dola | 0.0078 | 0.0640 | +0.0562 | 0.0609 | 0.1361 | +0.0752 |
| TriviaQA | vanilla_rag | csv_gated_vanilla_rag | 0.3750 | 0.3762 | +0.0012 | 0.4667 | 0.4672 | +0.0005 |
| TriviaQA | cad_fixed | csv_gated_cad | 0.3195 | 0.3423 | +0.0229 | 0.4108 | 0.4444 | +0.0336 |
| TriviaQA | acd | csv_gated_acd | 0.3523 | 0.2967 | -0.0556 | 0.4494 | 0.3944 | -0.0551 |
| TriviaQA | context_ucd | csv_gated_context_ucd | 0.2494 | 0.2999 | +0.0506 | 0.3415 | 0.4095 | +0.0680 |
| TriviaQA | dola | csv_gated_dola | 0.0279 | 0.1256 | +0.0977 | 0.0931 | 0.2223 | +0.1292 |

#### 0.4b best-tau summary

下表按 F1 为每个 `(model, dataset, baseline/plugin)` 选择最佳 tau：

| Model | dataset | Baseline | EM baseline | F1 baseline | Plugin | tau | EM plugin | F1 plugin | Delta EM | Delta F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma2-2B | NQ | vanilla_rag | 0.2576 | 0.3647 | csv_gated_vanilla_rag | 0.3 | 0.2598 | 0.3652 | +0.0022 | +0.0006 |
| Gemma2-2B | NQ | cad_fixed | 0.1701 | 0.2403 | csv_gated_cad | 0.5 | 0.2072 | 0.2915 | +0.0371 | +0.0512 |
| Gemma2-2B | NQ | acd | 0.2499 | 0.3592 | csv_gated_acd | 0.3 | 0.2033 | 0.3001 | -0.0465 | -0.0591 |
| Gemma2-2B | NQ | context_ucd | 0.1133 | 0.1694 | csv_gated_context_ucd | 0.5 | 0.1687 | 0.2441 | +0.0554 | +0.0747 |
| Gemma2-2B | NQ | dola | 0.0562 | 0.1440 | csv_gated_dola | 0.5 | 0.1114 | 0.2073 | +0.0551 | +0.0633 |
| Gemma2-2B | TriviaQA | vanilla_rag | 0.4329 | 0.5183 | csv_gated_vanilla_rag | 0.5 | 0.4926 | 0.5692 | +0.0598 | +0.0509 |
| Gemma2-2B | TriviaQA | cad_fixed | 0.3058 | 0.3899 | csv_gated_cad | 0.5 | 0.4217 | 0.5038 | +0.1159 | +0.1139 |
| Gemma2-2B | TriviaQA | acd | 0.4991 | 0.5760 | csv_gated_acd | 0.3 | 0.4949 | 0.5659 | -0.0042 | -0.0102 |
| Gemma2-2B | TriviaQA | context_ucd | 0.2029 | 0.2769 | csv_gated_context_ucd | 0.5 | 0.3601 | 0.4405 | +0.1573 | +0.1636 |
| Gemma2-2B | TriviaQA | dola | 0.1885 | 0.2890 | csv_gated_dola | 0.5 | 0.3425 | 0.4368 | +0.1541 | +0.1478 |
| Qwen3-4B | NQ | vanilla_rag | 0.2709 | 0.3636 | csv_gated_vanilla_rag | 0.3 | 0.2848 | 0.3780 | +0.0139 | +0.0144 |
| Qwen3-4B | NQ | cad_fixed | 0.2136 | 0.3082 | csv_gated_cad | 0.3 | 0.2446 | 0.3444 | +0.0310 | +0.0361 |
| Qwen3-4B | NQ | acd | 0.2806 | 0.3712 | csv_gated_acd | 0.3 | 0.2349 | 0.3249 | -0.0457 | -0.0463 |
| Qwen3-4B | NQ | context_ucd | 0.1460 | 0.2358 | csv_gated_context_ucd | 0.5 | 0.1945 | 0.2945 | +0.0485 | +0.0586 |
| Qwen3-4B | NQ | dola | 0.0078 | 0.0609 | csv_gated_dola | 0.5 | 0.0753 | 0.1505 | +0.0676 | +0.0896 |
| Qwen3-4B | TriviaQA | vanilla_rag | 0.3750 | 0.4667 | csv_gated_vanilla_rag | 0.3 | 0.3762 | 0.4672 | +0.0012 | +0.0005 |
| Qwen3-4B | TriviaQA | cad_fixed | 0.3195 | 0.4108 | csv_gated_cad | 0.3 | 0.3423 | 0.4444 | +0.0229 | +0.0336 |
| Qwen3-4B | TriviaQA | acd | 0.3523 | 0.4494 | csv_gated_acd | 0.3 | 0.2967 | 0.3944 | -0.0556 | -0.0551 |
| Qwen3-4B | TriviaQA | context_ucd | 0.2494 | 0.3415 | csv_gated_context_ucd | 0.5 | 0.3013 | 0.4110 | +0.0520 | +0.0695 |
| Qwen3-4B | TriviaQA | dola | 0.0279 | 0.0931 | csv_gated_dola | 0.5 | 0.1363 | 0.2343 | +0.1084 | +0.1412 |

结论：classifier plugin 对 CAD、Context-UCD、DoLA 的提升最稳定；对 vanilla RAG 的提升较小但在 Gemma2-2B TriviaQA 和 Qwen3-4B NQ 上为正；对 ACD 当前没有提升。DoLA baseline 在当前 HF DoLA 口径下较弱，已重跑 Gemma2-2B NQ 验证当前低分口径可复现。

### 0.5 还需要补的数据

| 优先级 | 待补内容 | 原因 |
|---|---|---|
| 中 | 是否跑 Gemma2-9B CSV_R sweep | 如果论文需要三模型 retrieved-doc CSV_R 完整性，需要补；当前已有 probe_R，但没有 CSV_R |
| 低 | 是否跑 Gemma2-9B pipeline | 资源成本高，建议等 2B/4B 结果稳定后再决定 |

## 一、项目总览

### 1.1 核心问题

在 RAG 系统中，检索到的文档不一定真正有助于回答问题。它可能是：

```text
relevant document：真正有助于回答问题的文档
distracting document：语义相关但不能帮助回答，甚至会误导模型的文档
```

本项目的核心目标是：**利用 LLM hidden states 判断检索文档是否有用，并进一步用这个判断结果控制后续解码。**

整个项目分为两个主要阶段：

```text
阶段 A：文档质量判断
训练 probe / CSV，在 LLM hidden states 上区分 relevant vs distracting document。

阶段 B：classifier plugin / gate
把训练好的 retrieved-doc classifier 接到已有 RAG 或 decoding baseline 前面；
当文档可信时启用 retrieved-doc 方法，当文档不可信时回退到 no-context / 更保守生成，从而减少坏检索带来的伤害。
```

---

### 1.2 三套评估口径

当前项目需要同时区分三套评估，避免把训练分布和 pipeline 分布混在一起：

| 评估类型 | 数据来源 | 指标 | 目的 |
|---|---|---|---|
| 旧分类器评估：gold-vs-distractor | `data/final/train.json` 和 `eval.json` | AUROC, Accuracy, Margin | 评估 hidden-state classifier / CSV 能否区分 gold evidence 与 distracting retrieved passage |
| 新分类器评估：retrieved-doc gate | `data/final_retrieved/train.json` 和 `eval.json` | AUROC, Accuracy, Margin | 评估模型能否判断真实 top-1 retrieved doc 是否值得信任；这是 Step11 gate 主线 |
| Pipeline 端到端评估 | `data/final/test_retrieval_nq.json` / `test_retrieval_triviaqa.json` | EM, F1 | 评估完整 RAG pipeline 的问答效果 |

核心变化：旧 Step9 CSV 仍然保留为 hidden-state separability / CSV amplification 证据；但 Step11 的文档 gate 不再使用旧 gold-vs-distractor CSV，而是使用新的 Step9R retrieved-doc CSV。

---

### 1.3 当前主实验模型

| 模型 | 参数量 | 类型 | 状态 | 作用 |
|---|---:|---|---|---|
| `google/gemma-2-2b` | 2B | base | 旧 Step9 best=`inject_1_cls_16`，AUROC=0.9964；新 Step9R retrieved-doc gate best=`inject_14_cls_18`，AUROC=0.9029，Acc=0.8375 | 主模型；retrieved-doc gate checkpoint 已可用于 score-only 与 gated pipeline |
| `google/gemma-2-9b` | 9B | base | 旧 Step9 best=`inject_0_cls_30`，AUROC=0.9974；retrieved-doc probe_R 已完成；CSV_R / pipeline 暂缓 | 等 2B / 4B 路线稳定后再 scale-up |
| `Qwen/Qwen3-4B-Base` | 4B | base | 旧 CSV Step9b 已完成；新 Step9R expanded + 固定 cls22 hparam grid 已完成；当前 CSV_R best=`inject_21_cls_22`，AUROC=0.9159；pipeline 已在 5090 跑完并同步到 `autodl_5090_2/results/pipeline/` | 跨架构 retrieved-doc gate 对照 |

LLaMA-3.1-8B 已下线，仅作为诊断记录保留。

> **关于 Qwen 模型选型**：当前主实验固定使用 `Qwen/Qwen3-4B-Base`，目的是与 Gemma2-base 系列保持 base-model 对齐。其他 Qwen 变体暂不纳入主实验，避免把 post-training / 架构差异混入当前对比。

---

### 1.4 全局进度总表

图例：`[✓]` 已完成 · `[~]` 进行中 · `[ ]` 未开始 · `[-]` 暂停 / 保留

```text
第一阶段:数据准备
  [✓] Step 1-5: 旧 gold-vs-distractor train/eval 与 pipeline test_retrieval 文件全部生成
  [✓] Step 4b:  新 retrieved-doc matched train/eval 数据生成完成；train 平衡，eval 自然分布

第二阶段 A: hidden state 分析与旧 gold-vs-distractor 分类器训练
  [✓] Step 6:    提取 hidden states (gemma2b / gemma9b / qwen3_4b)
  [✓] Step 6b:   验证 L2 norm 分布
  [✓] Step 7:    PCA 可视化
  [✓] Step 8:    LR / MLP / NearestCentroid / Mass-Mean probe (三模型 × 四 classifier)
  [✓] Step 8:    CRAG retrieval evaluator baseline 已完成，结果 `results/probe/crag_gold_distractor_baseline_results.json`
  [✓] Step 8b:   DPR encoder-dot baseline (gold-vs-distractor combined AUROC=0.2919，反相关；结果 `results/probe/dpr_baseline.json`)
  [✓] Step 9:    Gemma2-2B 旧 CSV sweep + Step9b 完成（best AUROC≈0.9964）
  [✓] Step 9:    Gemma2-9B 旧 CSV current sweep + Step9b 完成（38 configs；best AUROC≈0.9974）
  [✓] Step 9:    Qwen3-4B 旧 CSV current sweep + Step9b 完成（24 configs；best AUROC≈0.9961）

第二阶段 B: 新 retrieved-doc matched 分类器（Step11 gate 主线）
  [✓] Step 6R:   retrieved-doc hidden states 已新增并完成三模型提取（Gemma2B / Gemma9B / Qwen3-4B）
  [✓] Step 7R:   retrieved-doc PCA 可视化已完成三模型输出到 results/pca_retrieved/
  [✓] Step 8R:   retrieved-doc LR / MLP / Centroid / Mass-Mean probe 已完成三模型输出到 results/probe_retrieved/
  [✓] Step 8R:   DPR retriever-score baseline 已完成，使用 `data/final_retrieved/{train,eval}.json` 中 `doc.score`，结果 `results/probe_retrieved/dpr_baseline.json`
  [✓] Step 8R:   CRAG retrieval evaluator baseline 已完成，结果 `results/probe_retrieved/crag_baseline_results.json`
  [✓] Step 8R:   Self-RAG retrieval evaluator baseline 已完成，结果 `results/probe_retrieved/selfrag_baseline_results.json`
  [✓] Step 8:    Self-RAG gold-vs-distractor baseline 已完成，输出 `results/probe/selfrag_gold_distractor_baseline_results.json`
  [✓] Step 9R:   retrieved-doc CSV 训练脚本 step9_train_csv_retrieved.py 完成，并固定 seed=42
  [✓] Step 9R:   Gemma2B fixed-layer 调参完成：inject=1, cls=16, lr=0.002, ema=0.99 最好
  [✓] Step 9R:   Gemma2B expanded layer sweep 已完成：inject=1,2,3,4,5,6,10,12,16；cls=8,12,16,18,20,24,-1；共 57 个有效 config
  [✓] Step 9R:   Gemma2B 固定 CLS=18 inject sweep 已补跑：cls=18；inject=0..17；该层切片共 18 个 config（若接在 57-config sweep 后运行，新增约 9 个 unique config）
  [✓] Step 9R:   Gemma2B retrieved-doc gate best 已确认：`inject_14_cls_18`，AUROC=0.902907，Acc=0.837471，best_epoch=7；checkpoint=`results/csv_retrieved/gemma2b_retrieved_inject_14_cls_18_csv.pt`
  [✓] Step 9R:   Qwen3-4B fixed-layer 调参完成：固定 inject=4, cls=30 时，lr=0.002, ema=0.99 最好
  [✓] Step 9R:   Qwen3-4B expanded layer sweep 已完成：inject=2,4,6,8,10,11；cls=8,12,16,20,22,23,24,27,30,-1；共 57 个有效 config
  [✓] Step 9R:   Qwen3-4B 固定 cls=22 inject sweep 与 hparam grid 已完成：当前 best=`inject_21_cls_22`，AUROC=0.915867，Acc=0.855269，best_epoch=20
  [✓] Step 14:   retrieved-doc 2×2 ablation 已补齐；新增 `(c) CSV + LR head` 结果在 `c_ablation_retrieved_lr_20260513/results/ablation_csv_nonprototype/`
  [✓] Step 15A:  Cross-Encoder answer-support baseline 已完成；正式台账保留 gold-vs-distractor 与 retrieved-doc 两个公平 `answer_mode=none` 设置，结果在 `results/cross_encoder_gold/` 与 `results/cross_encoder_retrieved/`
  [✓] Step 15B:  Low-rank Representation Head baseline 已完成；Gemma2B/Qwen3-4B 的 gold 与 retrieved 两个口径结果在 `results/lowrank_rep_head_gold/` 与 `results/lowrank_rep_head_retrieved/`
  [ ] Step 9R:   Gemma2-9B retrieved-doc CSV sweep 待 2B / 4B 路线稳定后再跑

第三阶段:Logit Lens 与 classifier-gated pipeline
  [✓] Step 10:   Logit Lens 分析完成 (三模型 eval.json 全量 736 samples)
  [✓] Step 10b:  Logit Lens 画图完成（三模型 mean_logP / ΔlogP / subset 曲线均已生成）
  [✓] Step 10R:  retrieved-doc Logit Lens 已新增；Gemma2B/NQ 完成，retrieved relevant peak layer=23；当前作为解释 classifier-gated pipeline 的诊断证据，不作为新解码主线
  [✓] Step 11:   greedy-only pipeline generation 已更新；baseline 方法跳过 CSV；gated 方法仍使用 Step9R retrieved-doc CSV
  [✓] Step 11:   外部 decoding baseline 已接入并 smoke test 通过：no_doc / vanilla_rag / cad_fixed / acd / context_ucd / dola；`context_ucd` 已改为 energy-weighted UCD-style context contrast
  [✓] Step 11:   Gemma2B × 两数据集 × 6 baseline 已有正式 Step12 summary
  [✓] Step 11:   Gemma2B pipeline 已按固定 `alpha=0.5` 完成 5 个 classifier-plugin 对照，并已补齐 tau=0.5/0.4/0.3
  [✓] Step 11:   Qwen3-4B pipeline 已在 5090 服务器按同一口径完成，结果已同步到 `autodl_5090_2/results/pipeline/`

第四阶段:端到端评估
  [✓] Step 12:   评测脚本已实现，并修复 answers 字符串 list 解析问题
  [✓] Step 12:   已接入批量脚本，随 Step11 输出自动计算 EM/F1 并更新 summary
  [✓] 主实验表 + Ablation 表：Gemma2B 与 Qwen3-4B 已按 tau=0.5/0.4/0.3 整理，完整表在 `results/analysis/pipeline_plugin_comparison_by_tau.md`
  [✓] 结果整理:  gold-vs-distractor、retrieved-doc CSV_R、Gemma2B/Qwen3-4B pipeline 主表已写入本计划；可继续导出论文表格
```

> 说明：旧 Step9 的 Gemma2B final 结果仍作为 separability 主证据；新 Step9R 的结果将作为 Step11 gate 和最终 pipeline 的主输入。

---

#### 当前双主线说明（v8 新增）

```text
实验线 A：gold-vs-distractor CSV（旧线）
  数据：data/final/train.json / eval.json
  正类：gold relevant_doc
  负类：retrieved_top1 中被 GPT 标注为 distracting 的文档
  作用：证明 LLM hidden states / CSV 能强力区分 gold evidence 与 distracting retrieved passage
  状态：保留为 hidden-state separability / CSV amplification 证据；不作为 Step11 gate 输入

实验线 B：retrieved-doc CSV（新线）
  数据：data/final_retrieved/train.json / eval.json
  正类：retrieved_top1 被 GPT 标注为 relevant
  负类：retrieved_top1 被 GPT 标注为 distracting
  作用：判断真实 RAG pipeline 中 top-1 retrieved_doc 是否值得信任
  状态：作为 Step11 gate 主线；Gemma2B Step9R best 已确认并已用于 gated pipeline，Qwen3-4B CSV_R best 已确认并已整理为 pipeline 专用目录
```

为什么要新增实验线 B：旧 CSV 的正类是 gold evidence，而 Step11 真实输入是 DPR/FAISS top-1 retrieved doc，两者训练分布不一致。Step9R 直接在 retrieved top-1 分布上训练，更适合作为 pipeline gate。

---

#### 阶段二核心结论（已落地）

**1. 旧 gold-vs-distractor 口径下，DPR encoder dot 与 relevance 反相关**

```text
Combined: AUROC_raw=0.2919 (反相关), AUROC_aligned=0.7081
paired_acc=0.145 — 85% 的 (rel, dis) 对中 distract 分数更高
margin_mean(rel - dis) = -5.0459
```

原因：旧线的正类 `relevant_doc` 是 gold evidence，没有保存 DPR/FAISS retrieval score；负类 `distracting_doc` 是 DPR/FAISS top-1 retrieved passage。为了公平，Step8b 对两类都重新计算 DPR question/context encoder dot。结果仍然显示 distract docs 在 DPR encoder 空间中更像 query。**这是一个数据集构造特性，不是 bug**，且对论文叙事极为有利。

**2. LLM hidden states 揭示完全不同的信号**

```text
Step 8 best AUROC / Accuracy (combined):
              LR AUROC/Acc      MLP AUROC/Acc     Centroid AUROC/Acc   Mass-Mean AUROC/Acc   最佳层
  Gemma2-2B   0.8029 / 0.7446   0.8568 / 0.7779  0.7458 / 0.6855    0.7456 / 0.6875      layer 5
  Gemma2-9B   0.7605 / 0.7052   0.7702 / 0.7052  0.7415 / 0.6712    0.7404 / 0.6698      LR/Centroid/Mass-Mean layer 29, MLP layer 31
  Qwen3-4B    0.8127 / 0.7486   0.8325 / 0.7636  0.7768 / 0.7154    0.7777 / 0.7154      layer 23
```

MLP 通常强于 LR，而 Centroid 与 Mass-Mean 基本接近；这说明简单均值方向已捕获一部分信号，但非线性 probe 仍有额外收益。

**3. CSV 进一步把可分性推到极致**

```text
Step9 CSV 当前结果：

模型        状态       完成度/当前记录     best config             AUROC     Acc      Epoch
Gemma2-2B   final      72 configs       inject=1, cls=16     0.9964    0.9592   16
Gemma2-9B   final*     38 configs       inject=0, cls=30     0.9974    0.9660   6
Qwen3-4B    final*     24 configs       inject=6, cls=24     0.9961    0.9552   9
```

Gemma2-2B CSV final:
```text
AUROC=0.9964  Acc=0.9592  config: inject=1, cls=16, epoch=16
→ 比 MLP probe (0.8568) 提升约 0.14, 比 DPR encoder-dot raw AUROC (0.29) 提升约 0.71
```

Step9b 重评 (best ckpt 分 subset):
```text
Gemma2-2B:
  combined AUROC=0.9964  Acc=0.9592
  NQ       AUROC=0.9970  Acc=0.9614
  TriviaQA AUROC=0.9957  Acc=0.9575

Gemma2-9B:
  combined AUROC=0.9974  Acc=0.9654
  NQ       AUROC=0.9983  Acc=0.9722
  TriviaQA AUROC=0.9962  Acc=0.9600

Qwen3-4B:
  combined AUROC=0.9961  Acc=0.9552
  NQ       AUROC=0.9966  Acc=0.9583
  TriviaQA AUROC=0.9956  Acc=0.9527
```

这说明旧 gold-vs-distractor CSV 在 NQ 与 TriviaQA 上都非常稳定，明显强于静态 probe，并且学到的是跨 dataset 的 relevant/distract 判别方向。

注意：Gemma2-9B 当前 sweep 汇总包含 38 个 config，Qwen3-4B 当前 sweep 汇总包含 24 个 config；如果论文中要声称“完整原计划网格”，需补齐原计划剩余 config。

**核心论文叙事**：旧 gold-vs-distractor 口径下 DPR encoder dot (0.29) 与 relevance 反相关 → LLM 静态 hidden states (0.83-0.86) 已包含强信号 → CSV 通过可学习的轻量扰动把可分性放大到 (0.99+)，且该信号**跨 dataset 域无关**。这表明 LLM 在 generation 时对文档质量做出了 retrieval-time 信号无法做到的判断，且这种判断可以用一个简单可训练向量主动放大。

---

#### 阶段三 Step 10 核心结论（已完成）

Step 10 在 `data/final/eval.json` 的完整 736 条样本上，对 `no_ctx / gold_ctx / dis_ctx` 三种条件进行 teacher-forcing logit lens 分析。核心结论：**三个模型均满足 `gold_ctx > no_ctx ≈ dis_ctx`，说明 relevant document 会稳定增强正确答案信号，而 distracting document 不会产生类似增益。**

| 模型 | hidden states 数 | final no_ctx P | final gold_ctx P | final dis_ctx P | final gold/no ratio | Δgold-no peak layer | peak Δgold-no | final Δdis-no |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma2-2B | 27 | 0.0458 | 0.1134 | 0.0447 | 2.48× | 23 | +2.4700 | -0.0232 |
| Gemma2-9B | 43 | 0.0684 | 0.1273 | 0.0695 | 1.86× | 36 | +1.5295 | +0.0168 |
| Qwen3-4B | 37 | 0.0507 | 0.1602 | 0.0450 | 3.16× | 31 | +3.4918 | -0.1185 |

解释：
- Gemma2-9B 的 no-context 先验更强，因此 gold/no 相对增益小于 Gemma2-2B 和 Qwen3-4B。
- Qwen3-4B 对 gold context 最敏感，gold/no ratio 和 peak Δgold-no 均最高。
- 三个模型的 Δgold-no peak 都出现在后层相似相对位置：Gemma2-2B layer 23/26，Gemma2-9B layer 36/42，Qwen3-4B layer 31/36，约为模型深度的 86%-88%。这说明 gold document 带来的答案信号主要在后层形成，是后续 DoLA-style / contrastive decoding layer selection 的直接依据。
- `P≈exp(mean logP)` 表示答案 token 的平均概率强度，不是 EM/F1 或准确率。

实现校验：
- Gemma2 系列使用 `final_logit_softcapping=30.0`，Qwen3-4B 无 softcap。
- 已修正 final-layer double norm 问题：中间层 logit lens 使用 `final_norm(hs)`，最后一层 `hidden_states[-1]` 不再重复 final norm。
- 三个模型的 final-layer sanity check 均通过：`max abs diff=0.00000000`，`argmax match=True`，说明 logit lens final layer 能完全复现真实 `model(...).logits`。

---

### 1.5 硬件建议

| 模型 | 推荐硬件 | 说明 |
|---|---|---|
| Gemma2-2B | 1× RTX 4090 即可 | Step9 较快跑完 |
| Gemma2-9B | A800 / H800 / A100 80GB / Pro 6000 | 单张 4090 可能 CPU offload |
| Qwen3-4B-Base | 1× RTX 4090 (~8GB fp16) | 单卡 batch=4 稳定 |

---

## 二、项目结构（带用途说明）

这一节用于快速理解代码、数据和结果文件之间的关系。整体流程是：

```text
原始 QA 数据 / wiki_dpr
  → Step1 抽取 query + gold document
  → Step2 用 DPR/FAISS 检索 top-1 候选文档
  → Step3 用 GPT 标注哪些 top-1 是 distracting
  → Step4 构造 train/eval 分类数据
  → Step6 提取静态 hidden states，Step8 训练 probe，Step9 用原始 prompt 重新 forward 训练 CSV
  → Step10 做 logit lens 诊断
  → Step11-12 做 classifier-gated pipeline 和端到端评估
```

---

### 2.1 目录总览

```text
rag-probe/
├── config/                                   # 全局配置目录，集中管理路径、模型名、数据集名等常量
│   ├── __init__.py                           # 让 config 成为 Python package
│   └── settings.py                           # 全局路径与超参配置，例如 DATASETS / DPR_DIR / RETRIEVAL_TOP_K
│
├── scripts/                                  # 所有实验主脚本，按 step 编号对应实验流程
│   │
│   │  # ===== 第一阶段：数据准备 =====
│   ├── step1_extract_queries.py              # 从 NQ / TriviaQA 中抽取 question、answers、gold/relevant_doc
│   ├── step2_retrieve.py                     # 使用本地 wiki_dpr + FAISS，对每个 query 检索 DPR top-1 文档
│   ├── step3_annotate.py                     # 用 GPT 标注 retrieved top-1 是否是 distracting document
│   ├── step4_build_dataset.py                # 根据标注结果筛选 distracting，并构造旧 gold-vs-distractor train.json / eval.json
│   ├── step4b_build_retrieved_dataset.py      # 构造新 retrieved-doc matched 数据：doc=top1 retrieved, label=relevant/distracting
│   ├── step5_prepare_test.py                 # 构造 pipeline 端到端测试集 test_retrieval_nq/triviaqa.json
│   │
│   │  # ===== 第二阶段：hidden state 分析 & probe / CSV =====
│   ├── step6_extract_hidden_states.py         # 对 train/eval 的 relevant/distracting prompts 提取各层 last-token hidden states
│   ├── step6b_verify_norms.py                 # 检查 hidden state L2 norm 分布，判断 cosine/vMF 假设是否合理
│   ├── step7_pca_visualization.py             # 对 hidden states 做 PCA 可视化，定性观察 rel/dis 分布
│   ├── step8_train_probe.py                   # 训练 LR / MLP / NearestCentroid / Mass-Mean probe，评估 old gold-vs-distractor hidden states 可分性
│   ├── step6_extract_hidden_states_retrieved.py # Step6R：对 retrieved-doc matched 数据重新提取 hidden states
│   ├── step6b_verify_norms_retrieved.py       # Step6bR：retrieved-doc hidden states norm 验证
│   ├── step7_pca_visualization_retrieved.py   # Step7R：retrieved-doc PCA 可视化，输出到 results/pca_retrieved/
│   ├── step8_train_probe_retrieved.py         # Step8R：retrieved-doc LR / MLP / Centroid / Mass-Mean probe，输出到 results/probe_retrieved/
│   ├── step8_crag_baseline.py                 # Step8R baseline：CRAG retrieval evaluator，对 retrieved-doc relevance 做外部模型对照
│   ├── step8_selfrag_baseline.py              # Step8R baseline：Self-RAG [Relevant]/[Irrelevant] logprob evaluator
│   ├── step8b_dpr_baseline.py                 # 旧 gold-vs-distractor：计算 DPR encoder dot baseline，验证 DPR 分数是否能区分 rel/dis
│   ├── step8b_dpr_baseline_retrieved.py       # retrieved-doc：使用 doc.score / retriever_score 作为 DPR baseline
│   ├── step9_train_csv.py                     # 通用 CSV 训练入口：训练 Context Separator Vector
│   ├── step9_train_csv_gemma9b.py             # Gemma2-9B 专用 wrapper / 启动入口
│   ├── step9_train_csv_qwen3_4b.py            # Qwen3-4B 专用 wrapper / 启动入口
│   ├── step9_train_csv_retrieved.py           # Step9R：retrieved-doc CSV 训练入口，服务 Step11 gate
│   │
│   │  # ===== Step9 辅助运行脚本 =====
│   ├── run_step9_dual_gpu.sh                  # Gemma2-2B 旧 gold-vs-distractor CSV sweep shell 脚本（实际存在）
│   ├── run_step9_gemma9b.sh                   # Gemma2-9B CSV sweep shell 脚本，用于 AUTODL / 大显存机器
│   ├── run_step9_qwen3_4b.sh                  # Qwen3-4B CSV sweep shell 脚本
│   ├── run_step9r_gemma2b.sh                  # Gemma2B retrieved-doc CSV expanded sweep 脚本
│   ├── run_step9r_gemma2b_cls18_inject_sweep.sh # Gemma2B 固定 cls=18 的 inject 补扫脚本
│   ├── run_step9r_qwen3_4b.sh                 # Qwen3-4B retrieved-doc CSV expanded sweep 脚本
│   ├── run_step9r_gemma9b.sh                  # Gemma9B retrieved-doc CSV sweep 脚本（待迁移到 AUTODL）
│   │
│   │  # ===== 第三阶段：Logit Lens & classifier-gated pipeline =====
│   ├── step10_logit_lens.py                   # 对 no/gold/dis 三种上下文做逐层 logit lens，找答案信号 peak layer
│   ├── step10b_plot_logit_lens.py             # 读取 step10 json，画 mean_logP / Δgold-no / Δdis-no 曲线图
│   ├── step11_contrastive_decoding.py         # Pipeline 生成：greedy-only baseline + gated 方法；baseline 跳过 CSV，gated 方法读取 retrieved-doc CSV
│   ├── step14_csv_nonprototype_ablation.py    # 2×2 ablation 的 (c)：注入 CSV 后训练 LR/MLP 非 prototype head
│   ├── run_step11_greedy_baselines_with_eval.sh # 批量跑 6 个 baseline，并在每个数据集后自动调用 Step12 评测
│   │
│   │  # ===== 第四阶段：端到端评估 =====
│   └── step12_evaluate_pipeline.py            # 读取 Step11 generation JSON，计算 NQ / TriviaQA 的 EM、F1
│
├── csv_module/                                # CSV 训练所需的模型层 wrapper 与训练工具
│   ├── __init__.py                            # package 初始化
│   ├── llm_layers.py                          # 包装 Gemma2/Qwen3 decoder layer，在指定层注入 CSV 向量
│   └── train_utils.py                         # CSV loss、centroid、AUROC/Acc 评估等训练辅助函数
│
│   # 注：当前仓库没有 TSV/ 目录；TSV 只作为方法背景参考，不作为当前 repo 结构记录。
│
├── data/                                      # 所有输入数据、中间数据、最终训练数据和 hidden states
│   ├── dpr_download/                          # 本地 wiki_dpr 语料和 FAISS index，用于 Step2 检索
│   ├── queries/                               # Step1 输出：从 NQ/TQA 抽出的 query + gold document
│   ├── retrieved/                             # Step2 输出：每个 query 的 DPR/FAISS top-1 检索文档
│   ├── annotated/                             # Step3 输出：GPT 对 retrieved_top1 的 relevant/distracting 标注结果
│   ├── final/                                 # Step4/5 输出：旧 gold-vs-distractor train/eval 与 pipeline 测试集
│   ├── final_retrieved/                       # Step4b 输出：新 retrieved-doc matched train/eval，服务 Step9R/Step11 gate
│   └── hidden_states/                         # Step6 输出：三模型的 train/eval hidden states 张量
│       ├── gemma2b/                           # Gemma2-2B hidden states
│       ├── gemma9b/                           # Gemma2-9B hidden states
│       └── qwen3_4b/                          # Qwen3-4B hidden states
│
├── results/                                   # 所有实验结果、图和 checkpoint
│   ├── norms/                                 # Step6b 输出：norm 分布图和统计 json
│   ├── pca/                                   # Step7 输出：PCA 可视化图
│   ├── probe/                                 # Step8/8b 输出：旧 gold-vs-distractor probe 结果、DPR baseline、AUROC 曲线
│   ├── probe_retrieved/                       # Step8R 输出：retrieved-doc probe 结果和 AUROC by layer 图
│   ├── pca_retrieved/                         # Step7R 输出：retrieved-doc PCA 图
│   ├── norms_retrieved/                       # Step6bR 输出：retrieved-doc hidden-state norm 图和统计
│   ├── csv/                                   # 旧 Step9 输出：gold-vs-distractor CSV checkpoint、sweep json
│   ├── csv_retrieved/                         # 新 Step9R 输出：retrieved-doc CSV checkpoint、sweep json
│   ├── csv_retrieved_tune_qwen_i4_c30/        # Qwen3-4B 固定 best layer 的 lr/ema 调参结果
│   └── logit_lens/                            # Step10/10b 输出：logit lens json 和曲线图
│
├── third_party/                               # 第三方本地代码缓存，避免运行时访问 HuggingFace Hub
│   └── transformers-community-dola/           # 本地 DoLA custom generate 文件；通过 DOLA_CUSTOM_GENERATE 指定 generate.py
├── DPR/                                       # DPR 相关代码、下载脚本或本地检索工具
└── PLAN.md                                    # 项目计划、实验记录、当前结论和下一步
```

---

### 2.2 `data/` 目录详细说明

`data/` 是整个项目最重要的目录，因为后续所有 probe、CSV、logit lens 和 pipeline 评估都依赖这些文件。

| 路径 | 由哪一步生成 | 主要内容 | 后续用途 |
|---|---|---|---|
| `data/dpr_download/` | 手动下载 / Step2 前准备 | `wiki_dpr` passages + FAISS index | Step2 检索 top-1 候选文档 |
| `data/queries/` | Step1 | NQ / TriviaQA 的 `question`、`answers`、`relevant_doc` | Step2 的检索输入；保留 gold document 来源 |
| `data/retrieved/` | Step2 | 每个 query 的 `retrieved_top1`，通常包含 `text/title/score/passage_id` | Step3 标注输入；用于构造 distracting_doc |
| `data/annotated/` | Step3 | GPT 对 `retrieved_top1` 的标注，判断它是否 distracting | Step4 筛选高质量 distracting 样本 |
| `data/final/train.json` | Step4 | 训练集：2940 samples；每个 sample 包含 question、answers、relevant_doc、distracting_doc、annotation、source_dataset | Step6 从它构造 rel/dis prompts 并提取 train hidden states；Step9 直接读取它并重新 forward 训练 CSV |
| `data/final/eval.json` | Step4 | 评估集：736 samples；schema 与 train.json 一致 | Step6 从它构造 rel/dis prompts 并提取 eval hidden states；Step9 直接读取它并重新 forward 做 CSV eval；Step10 直接读取它做 no/gold/dis logit lens |
| `data/final_retrieved/train.json` | Step4b | retrieved-doc 训练集：5880 samples；doc 为 retrieved top-1；label 为 relevant/distracting；训练集按 dataset 内类别平衡 | Step9R retrieved-doc CSV 训练；服务 Step11 gate |
| `data/final_retrieved/eval.json` | Step4b | retrieved-doc 评估集：2135 samples；保持自然分布，combined relevant_rate≈0.655 | Step6R/8R retrieved probe 评估；Step9R retrieved-doc CSV 评估和 checkpoint 选择 |
| `data/hidden_states_retrieved/{model}/train_hidden_states.pt` | Step6R | retrieved-doc 训练集的各层 last-token hidden states；三模型已生成 | Step6bR norm；Step7R PCA；Step8R retrieved probe 训练 |
| `data/hidden_states_retrieved/{model}/eval_hidden_states.pt` | Step6R | retrieved-doc eval 的各层 last-token hidden states；三模型已生成 | Step6bR norm；Step7R PCA；Step8R retrieved probe 评估 |
| `data/final/test_retrieval_nq.json` | Step5 | NQ pipeline 测试集，每个 query 配 top-1 retrieved doc | Step11/12 端到端生成与 EM/F1 评估 |
| `data/final/test_retrieval_triviaqa.json` | Step5 | TriviaQA pipeline 测试集，每个 query 配 top-1 retrieved doc | Step11/12 端到端生成与 EM/F1 评估 |
| `data/hidden_states/{model}/train_hidden_states.pt` | Step6 | 训练集 rel/dis prompts 的各层 last-token hidden states（静态缓存，不含 CSV 注入） | Step6b norm 验证；Step7 PCA；Step8 probe 训练 |
| `data/hidden_states/{model}/eval_hidden_states.pt` | Step6 | 评估集 rel/dis prompts 的各层 last-token hidden states（静态缓存，不含 CSV 注入） | Step6b norm 验证；Step7 PCA；Step8 probe 评估 |


**重要更正：Step9 与 `data/hidden_states/` 的关系**

```text
Step8 probe：直接读取 Step6 生成的静态 hidden states `.pt` 文件。
Step9 CSV：不直接读取这些 `.pt` 文件。
```

原因是 CSV 训练会在指定 transformer layer 注入可学习向量 `v`。一旦注入 `v`，后续 hidden states 会随 forward 重新计算，因此不能使用 Step6 提前保存的“未注入 CSV”的静态 hidden states。Step9 的真实输入是 `data/final/train.json` 和 `data/final/eval.json`，脚本会从中构造 prompts，在模型 forward 过程中注入向量并计算分类 loss / eval metrics。

---

### 2.3 数据是怎么一步步构建出来的

#### Step1：从原始 QA 数据中抽取 query 和 gold document

输入来自 NQ / TriviaQA。对每个样本保留：

```text
question              问题文本
answers               标准答案 / alias answers
relevant_doc          原始数据中真正支持答案的 gold document

注意：`source_dataset` 不是 Step1 输出字段，而是在 Step4 构造最终 train/eval 时加入。
```

输出到：

```text
data/queries/nq_queries.json
data/queries/triviaqa_queries.json
```

这里的 `relevant_doc` 不是 DPR 检索出来的，而是原始 QA 数据自带或预处理得到的 gold evidence。

---

#### Step2：用 DPR/FAISS 检索 top-1 候选文档

对 Step1 的每个 question，用 DPR question encoder 编码 query，然后在本地 `wiki_dpr` + FAISS index 里检索 top-1 passage。

输出到：

```text
data/retrieved/nq_retrieved.json
data/retrieved/triviaqa_retrieved.json
```

典型字段：

```text
question
answers
relevant_doc          Step1 保留下来的 gold document
retrieved_top1        DPR/FAISS 检索出的 top-1 passage，内部通常包含 text/title/score/passage_id

注意：`score` 和 `passage_id` 位于 `retrieved_top1` dict 内；`source_dataset` 仍不是 Step2 输出字段。
```

注意：`retrieved_top1` 不等于 `relevant_doc`。在本项目里，很多 `retrieved_top1` 是语义相关但不能回答问题的文档，因此后面会被标注为 `distracting_doc`。

---

#### Step3：用 GPT 标注 retrieved_top1 是否 distracting

Step3 读取 Step2 的 `retrieved_top1`，用 GPT 判断它相对于当前 question 和 answers 是否属于：

```text
helpful / relevant：能支持正确答案
distracting：语义相关，但不能支持正确答案，甚至会误导
```

输出到：

```text
data/annotated/nq_annotated.json
data/annotated/triviaqa_annotated.json
```

这一步的目的不是重新标注 gold document，而是从 DPR top-1 中筛出高质量 distracting documents。

---

#### Step4：构造 train/eval 分类数据

Step4 把每个样本整理成统一 schema：

```text
question
answers
relevant_doc           label=1 的文档，来自原始 QA gold evidence
distracting_doc        label=0 的文档，来自 DPR top-1 且经过 GPT 标注确认
annotation             Step3 对 retrieved_top1 的 GPT 标注结果，通常包含 label/confidence/rationale 等
source_dataset         nq 或 triviaqa
```

说明：`annotation` 会保留在 `train.json/eval.json` 中，便于后续追溯 distracting 样本来源与标注依据；但 Step6/Step8/Step9 的 prompt 构造不会把 `annotation` 放进模型输入。

输出：

```text
data/final/train.json     2940 samples
data/final/eval.json       736 samples
```

用于分类器时，每个 sample 会展开成两条 prompt：

```text
Document: relevant_doc
Question: q
Answer:
label = 1

Document: distracting_doc
Question: q
Answer:
label = 0
```

因此：

```text
train.json 2940 samples → 5880 prompts
eval.json   736 samples → 1472 prompts
```

这就是 Step6、Step8、Step9、Step10 的核心数据来源：Step6 会从这里生成静态 hidden states，Step8 读取这些静态 hidden states，Step9 直接读取 json 重新 forward，Step10 直接读取 eval.json 做 logit lens。

---

#### Step5：构造 pipeline 端到端测试数据

Step5 构造的是后续 Step11/12 用的 pipeline 测试集。它更接近真实 RAG 设置：

```text
输入 question
系统只拿到 retrieved_doc / top-1 doc
然后模型决定如何生成答案
最后用 EM/F1 评价生成结果
```

输出：

```text
data/final/test_retrieval_nq.json
data/final/test_retrieval_triviaqa.json
```

这部分数据不用于 probe/CSV 训练，而是用于最终端到端评估。

---

### 2.4 `results/` 目录详细说明

| 路径 | 由哪一步生成 | 内容 | 用途 |
|---|---|---|---|
| `results/norms/` | Step6b | 各模型各层 hidden state L2 norm 分布图和统计 json | 判断 cosine/vMF 假设是否合理 |
| `results/pca/` | Step7 | NQ / TriviaQA 的 rel/dis PCA 可视化图 | 定性展示 hidden states 是否有可分趋势 |
| `results/probe/` | Step8 / Step8b | LR/MLP/Centroid/Mass-Mean probe 结果、DPR encoder-dot baseline、CRAG gold-vs-distractor baseline、AUROC by layer 图 | 分类 baseline 与论文主表素材 |
| `results/probe_retrieved/` | Step8R / Step8bR | retrieved-doc LR/MLP/Centroid/Mass-Mean probe、DPR retriever-score baseline、CRAG baseline、Self-RAG baseline、AUROC by layer 图 | retrieved-doc gate 的分类 baseline 与论文主表素材 |
| `results/csv/` | Step9 / Step9b | 旧 gold-vs-distractor CSV checkpoint、单 config 结果、sweep 汇总、per-subset 重评结果 | 证明 CSV amplification；不再作为 Step11 gate 主线 |
| `results/csv_retrieved/` | Step9R | 新 retrieved-doc CSV checkpoint、单 config 结果、sweep 汇总 | Step11 classifier-plugin gate 输入 |
| `results/csv_retrieved_tune_qwen_i4_c30/` | Qwen3 Step9R 调参 | 固定 inject=4, cls=30 的 Qwen3 lr/ema 调参结果 | 选择 Qwen3 retrieved-doc expanded sweep 的超参 |
| `c_ablation_retrieved_lr_20260513/results/ablation_csv_nonprototype/` | Step14 | retrieved-doc 2×2 ablation 的 `(c) CSV + non-prototype LR head` 结果 | 与普通 LR、Centroid、Ours / CSV_R 组成 `CSV × Prototype` 消融表 |
| `results/logit_lens/` | Step10 / Step10b | 每层 mean_logP、Δgold-no、Δdis-no 的 json，以及 logP/ΔlogP/subset 曲线图 | 为 contrastive decoding 选择 layer / α 提供依据 |
| `results/pipeline/` | Step11 / Step12 | Step11 生成逐样本答案 JSON；Step12 读取这些 JSON 后生成 `*_eval.json`、`*_summary.csv` 等 EM/F1 评测文件 | 最终端到端实验主表和 ablation 表 |

---

### 2.4b 最终结果整理口径（规划中）

后续结果整理分成两类，每类都按模型拆成独立文件，避免把 classifier 结果和 pipeline 结果混在同一张表里。

**A. Classifier / CSV 结果文件：每个模型一个文件**

这一类文件汇总文档质量判断相关结果，包括：

```text
1. 旧 gold-vs-distractor 结果
   - Step8 LR / MLP / Centroid / Mass-Mean probe
   - Step8b DPR encoder-dot baseline
   - Step9 old CSV / Step9b per-subset

2. 新 retrieved-doc 结果
   - Step8R LR / MLP / Centroid / Mass-Mean probe
   - Step8bR DPR retriever-score baseline
   - Step9R retrieved-doc CSV sweep
   - fixed-layer / hparam sweep 的 best config

3. 必要的 baseline / 对照
   - DPR score baseline
   - static probe baselines
   - CSV vs probe 的提升
```

规划输出：

```text
results/analysis/gemma2b_classifier_results.md
results/analysis/gemma2b_classifier_results.csv

results/analysis/gemma9b_classifier_results.md
results/analysis/gemma9b_classifier_results.csv

results/analysis/qwen3_4b_classifier_results.md
results/analysis/qwen3_4b_classifier_results.csv
```

注意：`CSV_Retrieved` 不是替代旧 CSV，而是和旧 gold-vs-distractor CSV 并列展示。旧 CSV 证明 hidden-state separability / CSV amplification，新 Step9R retrieved-doc CSV 才是 Step11 gate 的实际输入。

**B. Pipeline 结果文件：每个模型一个文件**

这一类文件只整理端到端 QA 结果，包括：

```text
1. baseline:
   no_doc / vanilla_rag / cad_fixed / acd / context_ucd / dola

2. classifier-plugin methods:
   csv_gated_vanilla_rag / csv_gated_cad / csv_gated_acd / csv_gated_context_ucd / csv_gated_dola

3. per dataset metrics:
   NQ EM/F1
   TriviaQA EM/F1
   trusted_rate；主实验固定 CAD alpha=0.5、gate tau=0.5
```

规划输出：

```text
results/analysis/gemma2b_pipeline_results.md
results/analysis/gemma2b_pipeline_results.csv

results/analysis/gemma9b_pipeline_results.md
results/analysis/gemma9b_pipeline_results.csv

results/analysis/qwen3_4b_pipeline_results.md
results/analysis/qwen3_4b_pipeline_results.csv
```

当前状态：这两类 per-model 汇总文件暂未统一生成；已有原始结果仍保留在各自目录中：

```text
classifier raw/sweep:
  results/probe/
  results/probe_retrieved/
  results/csv/
  results/csv_retrieved/
  results/csv_retrieved_qwen3_4b_cls22_hparam_grid/

pipeline raw/summary:
  results/pipeline/
  results/pipeline/pipeline_eval_summary.csv
  results/all_finished_experiments.csv
```

---

### 2.5 最重要的数据流关系

```text
Step1/2/3/4 构造 data/final/train.json 和 eval.json
        ↓
Step6 把 train/eval 的 prompt 变成 hidden states
        ↓
Step8 用 Step6 缓存的静态 hidden states 训练普通 probe
Step9 不读取 Step6 的静态 hidden-state 缓存，而是直接读取 train/eval.json，构造 prompt，注入 CSV 向量并重新 forward；分类评估使用注入后的 hidden states 和训练得到的 centroids
        ↓
Step10 不训练，直接用 eval.json 做 teacher-forcing logit lens
        ↓
Step11 中 baseline 方法（`no_doc` / `vanilla_rag` / `cad_fixed` / `acd` / `context_ucd` / `dola`）不使用 classifier；plugin 方法（`csv_gated_vanilla_rag` / `csv_gated_cad` / `csv_gated_acd` / `csv_gated_context_ucd` / `csv_gated_dola`）使用 Step9R retrieved-doc CSV 判断 retrieved top-1 document 是否可信。Step8R retrieved-doc probe、CRAG baseline 和旧 gold-vs-distractor Step8/Step9 主要作为分析与对照证据，不作为当前 Step11 gate 的直接输入。

Step11 当前 `context_ucd` 是 Context-UCD-Energy：同模型下使用 `Document + Question` 与 `Question only` 两路 logits 的 UCD-style energy-weighted contrast，不是原始 expert-vs-amateur UCD。
        ↓
Step12 在 test_retrieval_nq/triviaqa.json 上评估 EM/F1
```

一句话理解：

```text
train/eval.json 是 Step6/8/9/10 的核心数据源：
- Step6/8 使用它们生成并读取静态 hidden states；
- Step9 直接读取它们重新 forward 训练 CSV；
- Step10 直接读取 eval.json 做 teacher-forcing logit lens。

test_retrieval_*.json 不参与 probe/CSV 训练，主要用于 Step11/12 验证真实 RAG 问答效果。
```

---

## 三、第一阶段：数据准备

### Step 1-5：构造 train/eval/test 数据集 [✓]

**做什么**：从 NQ / TriviaQA 抽取 query → DPR 检索 → GPT 标注 distracting → 拼成 train/eval/test。

**输出**：
```text
data/final/train.json                      2940 samples → 5880 prompts
data/final/eval.json                        736 samples → 1472 prompts
data/final/test_retrieval_nq.json
data/final/test_retrieval_triviaqa.json
```

**状态**：
- [✓] DPR NQ / TriviaQA train/test 数据准备
- [✓] wiki_dpr_nq + FAISS 索引
- [✓] step1-5 全部完成
- [✓] train/eval question overlap = 0

---

#### 数据构造逻辑（详细）

每个问题构造两条 prompt：

```text
Document: relevant_doc
Question: q
Answer:
label = 1

Document: distracting_doc
Question: q
Answer:
label = 0
```

`distracting_doc` 里可能有 `score`、`passage_id` 等额外字段，不能直接把 dict 转字符串塞入 prompt（会泄露捷径标签）。所有脚本统一用 `extract_doc_text()`：

```python
def extract_doc_text(doc):
    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip()
    if title:
        return f"Title: {title}\nText: {text}"
    return text
```

每个 sample 包含：

```text
question
answers
relevant_doc           来自原始 QA 数据的 gold document
distracting_doc        FAISS top-1 检索后, 经 GPT 标注为 distracting 的文档
annotation             GPT 标注信息，用于追溯 distracting 筛选依据，不进入 prompt
source_dataset         "nq" 或 "triviaqa"
```

分类器数据用于 Step6/8/9；其中 Step8 读取 Step6 生成的静态 hidden states，Step9 直接读取 train/eval json 并重新 forward。pipeline 测试集用于 Step11/12 端到端生成与 EM/F1 评估。

---

### Step 4b：构造 retrieved-doc matched 数据集 [✓]

**做什么**：把 Step3 中 GPT 对 `retrieved_top1` 的标注直接转成 pipeline-matched 分类数据。这里每条样本只有一个 `doc`，就是 DPR/FAISS top-1 retrieved document；标签来自 annotation：`relevant=1`，`distracting=0`。

**为什么新增 Step4b**：旧 Step4 的正类是 gold `relevant_doc`，负类是 retrieved distracting doc；但 Step11 真实 pipeline 里只看到 retrieved top-1。为了让 gate 的训练分布和 pipeline 输入一致，需要单独构造 `data/final_retrieved/`。

**当前构建命令**：
```bash
python scripts/step4b_build_retrieved_dataset.py
```

**当前输出**：
```text
data/final_retrieved/all.json
data/final_retrieved/train_unbalanced.json
data/final_retrieved/train.json
data/final_retrieved/eval.json
data/final_retrieved/stats.json
```

**当前分布**：
```text
ALL / natural:
  combined: total=10664  relevant=6988  distracting=3676  relevant_rate=0.655
  nq:       total= 5755  relevant=4192  distracting=1563  relevant_rate=0.728
  triviaqa: total= 4909  relevant=2796  distracting=2113  relevant_rate=0.570

TRAIN / used for training, balanced within dataset:
  combined: total=5880  relevant=2940  distracting=2940  relevant_rate=0.500
  nq:       total=2500  relevant=1250  distracting=1250
  triviaqa: total=3380  relevant=1690  distracting=1690

EVAL / natural:
  combined: total=2135  relevant=1399  distracting=736  relevant_rate=0.655
  nq:       total=1152  relevant=839   distracting=313
  triviaqa: total=983   relevant=560   distracting=423
```

**schema**：
```text
question
answers
doc                 # retrieved top-1 文档，只取 title/text 进 prompt
label               # 1=relevant, 0=distracting
label_name          # relevant / distracting
annotation          # GPT 标注信息，保留追溯，不进入 prompt
source_dataset      # nq / triviaqa
```

**用途**：Step9R 的唯一训练/评估数据源；Step11 gated 方法最终读取 Step9R checkpoint，对 test retrieved doc 计算 `P(relevant)`。

---

## 四、第二阶段：Hidden state 分析与分类器训练

第二阶段目标：找到最适合区分 relevant / distracting 的 hidden state 层、分类方式和模型。

主线模型：`gemma2b` / `gemma9b` / `qwen3_4b`。

---

### Step 6：提取 hidden states [✓]

**做什么**：对每个模型，对 train/eval 中每条 prompt 提取所有层的 last-token hidden state。

**命令**：
```bash
python scripts/step6_extract_hidden_states.py --model gemma2b  --batch_size 32
python scripts/step6_extract_hidden_states.py --model gemma9b  --batch_size 2
python scripts/step6_extract_hidden_states.py --model qwen3_4b --batch_size 4
```

**输出**：
```text
data/hidden_states/gemma2b/{train,eval}_hidden_states.pt
data/hidden_states/gemma9b/{train,eval}_hidden_states.pt
data/hidden_states/qwen3_4b/{train,eval}_hidden_states.pt
```

**状态**：
- [✓] gemma2b
- [✓] gemma9b
- [✓] qwen3_4b

---

#### 详细说明

存储格式：

```python
{
    "hidden_states": Tensor[N, L, D],
    "labels":        List[str],   # "relevant" / "distracting"
    "dataset_source": List[str],  # "nq" / "triviaqa" / "unknown"
    "questions":     List[str],
}
```

L = 1 + transformer 层数（包含 embedding 层）。

`MODEL_REGISTRY`（默认值改为 HF 名字，本地路径通过环境变量传入）：

```python
MODEL_REGISTRY = {
    "gemma2b":  {"hf_name": os.environ.get("GEMMA_MODEL_PATH",   "google/gemma-2-2b"),     "save_key": "gemma2b"},
    "gemma9b":  {"hf_name": os.environ.get("GEMMA9B_MODEL_PATH", "google/gemma-2-9b"),     "save_key": "gemma9b"},
    "qwen3_4b": {"hf_name": os.environ.get("QWEN3_4B_MODEL_PATH","Qwen/Qwen3-4B-Base"),    "save_key": "qwen3_4b"},
}
```

**Padding 选择**：Step6 用左 padding，所有序列右对齐，`hidden_states[:, -1, :]` 是 last token。Step9 用右 padding（匹配 TSV/CSV 注入），并跑 right-padding sanity check 验证 batch forward 与 single forward 一致。

**Qwen3-4B 在 Step6 没有兼容性问题**：只走 `AutoModelForCausalLM.forward()`，不涉及 CSV wrapper。验证项已通过：
- Step6 使用 `tokenizer.padding_side = "left"`，并通过 left-padding sanity check
- tokenizer.pad_token 的具体值以运行日志为准；代码只保证如无 pad token 会添加专用 `[PAD]`
- hidden_states 数量正确 (37 = 1 + 36)

---

### Step 6b：Embedding norm 验证 [✓]

**做什么**：验证各层 hidden state 的 L2 norm 是否集中，为 vMF / cosine-based 分类提供合理性。

**命令**：
```bash
python scripts/step6b_verify_norms.py --model all
```

**输出**：
```text
results/norms/{model}_norm_distributions.png
results/norms/{model}_norm_stats.json
```

**状态**：
- [✓] gemma2b
- [✓] gemma9b
- [✓] qwen3_4b

---

#### 详细说明

判断标准：

```text
CV < 0.1：norm 分布集中, cosine/vMF 假设比较合理
CV 明显较大：需要谨慎解释 CSV / centroid 结果
```

只读 step6 的 .pt 文件，几秒钟跑完。

---

### Step 7：PCA 可视化 [✓]

**做什么**：每个模型每层 hidden state 做 PCA 可视化，观察 relevant/distracting 的分布。

**命令**：
```bash
python scripts/step7_pca_visualization.py --model all
```

**输出**：
```text
results/pca/{model}_pca_nq.png
results/pca/{model}_pca_triviaqa.png
```

**状态**：
- [✓] gemma2b
- [✓] gemma9b
- [✓] qwen3_4b

---

#### 详细说明

PCA 只作为定性分析，不作为最终层选择依据。最终层选择以 Step8 的 AUROC 为准。

---

### Step 8：LR / MLP / NearestCentroid / Mass-Mean Probe [✓]

**做什么**：每个模型每层训练四种 probe，作为 CSV 之外的 baseline。共享 PCA → 16 维 → L2 归一化预处理。Mass-Mean 使用 `mean(relevant) - mean(distracting)` 作为无 bias 线性方向，区别于 nearest-centroid 的距离判别。

**命令**：
```bash
# 全部模型 × 全部 classifier (推荐)
python scripts/step8_train_probe.py --model all

# 单个模型
python scripts/step8_train_probe.py --model gemma2b
python scripts/step8_train_probe.py --model gemma9b
python scripts/step8_train_probe.py --model qwen3_4b

# 只跑某一种 classifier
python scripts/step8_train_probe.py --model qwen3_4b --classifier mlp
```

**输出**：
```text
results/probe/{model}_probe_results_lr.json
results/probe/{model}_probe_results_mlp.json
results/probe/{model}_probe_results_centroid.json
results/probe/{model}_probe_results_mass_mean.json
results/probe/{model}_auroc_by_layer.png       # classifier 同图对比
```

**状态**：
- [✓] gemma2b × {LR, MLP, Centroid, Mass-Mean}
- [✓] gemma9b × {LR, MLP, Centroid, Mass-Mean}
- [✓] qwen3_4b × {LR, MLP, Centroid, Mass-Mean}

**Best AUROC + Accuracy (combined subset)**：

| 模型 | LR AUROC | LR Acc | MLP AUROC | MLP Acc | Centroid AUROC | Centroid Acc | Mass-Mean AUROC | Mass-Mean Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma2-2B | 0.8029 | 0.7446 | **0.8568** | **0.7779** | 0.7458 | 0.6855 | 0.7456 | 0.6875 |
| Gemma2-9B | 0.7605 | 0.7052 | **0.7702** | 0.7052 | 0.7415 | 0.6712 | 0.7404 | 0.6698 |
| Qwen3-4B  | 0.8127 | 0.7486 | **0.8325** | **0.7636** | 0.7768 | 0.7154 | 0.7777 | 0.7154 |

---

#### 详细说明

| classifier | 实现 | 备注 |
|---|---|---|
| LR | `LogisticRegression(max_iter=1000, C=1.0)` | 强线性 baseline |
| MLP | `MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, early_stopping=True)` | 非线性 probe baseline |
| Centroid | 手动计算 L2-normalized PCA 空间中的 class centroids；用 `d0 - d1` 作为 AUROC score，`d1 < d0` 作为 relevant 预测 | 与 CSV 的 cosine/centroid 评估口径接近 |
| Mass-Mean | 手动计算 `direction = mean(relevant) - mean(distracting)`；用 `x @ direction` 作为 AUROC score，`score >= 0` 作为 relevant 预测 | 对齐论文 mass-mean probing 的轻量无优化方向 baseline |

MLP `(128,)` 通过 Gemma2-2B layer 12 的 grid search 选定，在 `{(32,), (64,), (128,), (32,16), (64,32)}` 中 mean AUROC 最高（0.6976）、std 最低（0.0019）。

Centroid probe 是手写 nearest-centroid：在 L2-normalized PCA 空间中分别计算 distracting/relevant centroid，距离差 `d_to_distracting - d_to_relevant` 越大表示越接近 relevant。由于 L2-normalized 空间中的欧式距离与 cosine geometry 对齐，因此它与 CSV 的 cosine/centroid 评估口径接近。

---

### Step 8R：Retrieved-doc Probe Baseline [✓]

**做什么**：在新的 pipeline-matched `data/final_retrieved/{train,eval}.json` 上重新提取 hidden states，并训练 LR / MLP / NearestCentroid / Mass-Mean probe。该结果用于回答：在真实 top-1 retrieved document 分布上，不注入 CSV 的静态 hidden states 本身能否判断 retrieved doc 是否值得信任。

**为什么不能复用旧 Step8**：旧 Step8 的正类是 gold `relevant_doc`，负类是 retrieved `distracting_doc`；Step11 真实输入只有 retrieved top-1。因此需要新的 hidden states 和 probe 结果。

**新增脚本**：
```text
scripts/step6_extract_hidden_states_retrieved.py
scripts/step6b_verify_norms_retrieved.py
scripts/step7_pca_visualization_retrieved.py
scripts/step8_train_probe_retrieved.py
scripts/step8_crag_baseline.py
scripts/step8b_dpr_baseline_retrieved.py
```

**输入 / 输出目录**：
```text
输入：data/final_retrieved/train.json / eval.json
hidden states：data/hidden_states_retrieved/{model}/{train,eval}_hidden_states.pt
probe：results/probe_retrieved/{model}_probe_results_{lr,mlp,centroid}.json
CRAG baseline：results/probe_retrieved/crag_baseline_results.json
PCA：results/pca_retrieved/{model}_pca_{nq,triviaqa}.png
norm：results/norms_retrieved/{model}_norm_*.json/png
```

**已运行命令**：
```bash
# Gemma2B
CUDA_VISIBLE_DEVICES=1 python scripts/step6_extract_hidden_states_retrieved.py --model gemma2b --batch_size 32
python scripts/step8_train_probe_retrieved.py --model gemma2b --classifier all
python scripts/step7_pca_visualization_retrieved.py --model gemma2b

# Qwen3-4B
CUDA_VISIBLE_DEVICES=0 python scripts/step6_extract_hidden_states_retrieved.py --model qwen3_4b --batch_size 4
python scripts/step8_train_probe_retrieved.py --model qwen3_4b --classifier all
python scripts/step7_pca_visualization_retrieved.py --model qwen3_4b

# Gemma2-9B
CUDA_VISIBLE_DEVICES=0 python scripts/step6_extract_hidden_states_retrieved.py --model gemma9b --batch_size 2
python scripts/step8_train_probe_retrieved.py --model gemma9b --classifier all
python scripts/step7_pca_visualization_retrieved.py --model gemma9b

# CRAG retrieval evaluator baseline: retrieved-doc gate 线
CUDA_VISIBLE_DEVICES=0 python scripts/step8_crag_baseline.py --mode retrieved

# CRAG retrieval evaluator baseline: 旧 gold-vs-distractor 线
CUDA_VISIBLE_DEVICES=0 python scripts/step8_crag_baseline.py --mode gold_distractor

# Self-RAG retrieval evaluator baseline: retrieved-doc gate 线
CUDA_VISIBLE_DEVICES=0 python scripts/step8_selfrag_baseline.py \
  --mode retrieved \
  --model_name /path/to/selfrag_llama2_13b \
  --batch_size 1 \
  --gpu_memory_utilization 0.80 \
  --cpu_offload_gb 4

# Self-RAG retrieval evaluator baseline: 旧 gold-vs-distractor 线
CUDA_VISIBLE_DEVICES=0 python scripts/step8_selfrag_baseline.py \
  --mode gold_distractor \
  --model_name /path/to/selfrag_llama2_13b \
  --batch_size 1 \
  --gpu_memory_utilization 0.80 \
  --cpu_offload_gb 4
```

**CRAG retrieval evaluator baseline（retrieved-doc 外部模型对照）**：

目的：用 CRAG 的 T5-based retrieval evaluator 直接判断 `question [SEP] passage` 是否 relevant，作为 Step8R/Step9R 之外的 retrieval evaluator baseline。

```text
代码：scripts/step8_crag_baseline.py
模型：CRAG/model
retrieved-doc 输入：data/final_retrieved/train.json / eval.json
retrieved-doc 输出：results/probe_retrieved/crag_baseline_results.json
retrieved-doc 当前结果：combined AUROC=0.5171，Acc=0.4244；NQ AUROC=0.4736；TriviaQA AUROC=0.5759

gold-vs-distractor 输入：data/final/train.json / eval.json
gold-vs-distractor 输出：results/probe/crag_gold_distractor_baseline_results.json
gold-vs-distractor 当前结果：combined AUROC=0.4519，Acc=0.5000；NQ AUROC=0.4393；TriviaQA AUROC=0.4631
```

**Self-RAG retrieval evaluator baseline（retrieved-doc 外部模型对照）**：

目的：用 Self-RAG 的 `[Relevant]` / `[Irrelevant]` token logprob 差值判断 top-1 retrieved document 是否 relevant。该 baseline 不使用 Gemma hidden states，作为外部 RAG-aware evaluator 对照。

```text
代码：scripts/step8_selfrag_baseline.py
模型：/path/to/selfrag_llama2_13b
retrieved-doc 输入：data/final_retrieved/train.json / eval.json
retrieved-doc 输出：results/probe_retrieved/selfrag_baseline_results.json
retrieved-doc 当前结果：combined AUROC=0.7688，Acc=0.6984；NQ AUROC=0.7669；TriviaQA AUROC=0.8263

gold-vs-distractor 输入：data/final/train.json / eval.json
gold-vs-distractor 输出：results/probe/selfrag_gold_distractor_baseline_results.json
gold-vs-distractor 当前结果：combined AUROC=0.6451，Acc=0.6046；NQ AUROC=0.6435；TriviaQA AUROC=0.6527
```

**retrieved-doc probe full table（best layer 口径，与旧 Step8 表一致）**：

填表口径：对每个 `(model, classifier)`，先用 `combined AUROC` 选 best layer/config；然后在同一 best layer/config 上填写 `combined / nq / triviaqa` 的 AUROC 与 Accuracy。`Ours / CSV` 对应 Step9R retrieved-doc CSV。Gemma2B 与 Qwen3-4B 的 CSV_R 子集数字均已由 validation JSON 回填；Gemma2-9B 的 retrieved-doc static probe baseline 已完成，但 CSV_R 仍未作为主线结果运行。

| 模型 | Classifier | Best layer/config | combined AUROC | NQ AUROC | TriviaQA AUROC | combined Acc | NQ Acc | TriviaQA Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| External | DPR retriever score | - | 0.6376 | 0.6042 | 0.6223 | 0.6253 | 0.6528 | 0.5931 |
| External | CRAG evaluator | - | 0.5171 | 0.4736 | 0.5759 | 0.4244 | 0.3385 | 0.5249 |
| External | Self-RAG evaluator | - | 0.7688 | 0.7669 | 0.8263 | 0.6984 | 0.6615 | 0.7416 |
| Gemma2-2B | LR | layer_18 | 0.8497 | 0.7996 | 0.8941 | 0.7536 | 0.7153 | 0.8199 |
| Gemma2-2B | MLP | layer_18 | 0.8486 | 0.7949 | 0.8925 | 0.7607 | 0.7005 | 0.8220 |
| Gemma2-2B | Centroid | layer_18 | 0.8403 | 0.7638 | 0.8779 | 0.7696 | 0.6502 | 0.7955 |
| Gemma2-2B | Mass-Mean | layer_18 | 0.8403 | 0.7637 | 0.8781 | 0.7756 | 0.6545 | 0.7976 |
| **Gemma2-2B** | **Ours / CSV (Step9R)** | **inject_14_cls_18** | **0.9029** | **0.8558** | **0.9379** | **0.8375** | **0.8116** | **0.8678** |
| Gemma2-9B | LR | layer_27 | 0.8893 | 0.8396 | 0.9321 | 0.8005 | 0.7587 | 0.8616 |
| Gemma2-9B | MLP | layer_29 | 0.8995 | 0.8379 | 0.9262 | 0.8108 | 0.7587 | 0.8576 |
| Gemma2-9B | Centroid | layer_27 | 0.8780 | 0.8238 | 0.9237 | 0.8108 | 0.7344 | 0.8566 |
| Gemma2-9B | Mass-Mean | layer_27 | 0.8781 | 0.8239 | 0.9238 | 0.8173 | 0.7457 | 0.8576 |
| **Gemma2-9B** | **Ours / CSV (Step9R)** | **not run yet** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| Qwen3-4B | LR | layer_23 | 0.8918 | 0.8587 | 0.9272 | 0.8262 | 0.7951 | 0.8678 |
| Qwen3-4B | MLP | layer_23 | 0.8950 | 0.8580 | 0.9180 | 0.8351 | 0.7899 | 0.8545 |
| Qwen3-4B | Centroid | layer_23 | 0.8857 | 0.8401 | 0.9161 | 0.8201 | 0.7726 | 0.8423 |
| Qwen3-4B | Mass-Mean | layer_23 | 0.8860 | 0.8402 | 0.9162 | 0.8220 | 0.7752 | 0.8423 |
| **Qwen3-4B** | **Ours / CSV (Step9R)** | **inject_21_cls_22** | **0.9159** | **0.8821** | **0.9409** | **0.8553** | **0.8403** | **0.8728** |

**Qwen3-4B Ours / CSV 子集指标补充命令（已运行）**：

Qwen3-4B 的 CSV_R best 已经跑完，但这张表还需要同一 best checkpoint 在 `data/final_retrieved/eval.json` 上按 NQ / TriviaQA 分开复算。本地运行：

```bash
cd /path/to/rag-probe

export QWEN3_4B_MODEL_PATH=Qwen/Qwen3-4B-Base
export CSV_RETRIEVED_DIR=/path/to/rag-probe/results/csv_retrieved_qwen3_4b_best_for_pipeline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHON_BIN=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=1 \
python scripts/step11_score_test_docs_only.py \
  --model qwen3_4b \
  --dataset all \
  --tau 0.5 \
  --batch_size 4
```

该命令会生成/更新：

```text
results/pipeline/qwen3_4b_retrieved_csv_eval_validation.json
results/pipeline_classifier_scores/qwen3_4b_test_retrieval_csv_score_summary.csv
```

说明：该命令不是旧 `scripts/step9b_reeval_per_subset.py`。旧 Step9b 读取 `data/final/eval.json` 和 `results/csv/`，用于 gold-vs-distractor；这里要补的是 retrieved-doc CSV_R，因此必须指定 `CSV_RETRIEVED_DIR`，并让 Step11 validation 读取 `data/final_retrieved/eval.json`。

当前已完成并确认 `best_key=inject_21_cls_22`、`passed=true`。复算结果：

```text
combined: AUROC=0.915867, Acc=0.855269
NQ:       AUROC=0.882094, Acc=0.840278
TriviaQA: AUROC=0.940928, Acc=0.872838
```

test retrieval score summary 也已生成：

```text
results/pipeline_classifier_scores/qwen3_4b_test_retrieval_csv_score_summary.csv
NQ:       trusted@0.5 = 1797/3610  = 0.4978, P(relevant) mean=0.4895
TriviaQA: trusted@0.5 = 4744/11313 = 0.4193, P(relevant) mean=0.4288
```

**简表观察**：

```text
1. retrieved-doc probe 的最佳层集中在中后层：Gemma2B layer 18，Gemma9B layer 27/29，Qwen3-4B layer 23。
2. 该任务比旧 gold-vs-distractor 更贴近 Step11 pipeline gate，也更真实；静态 probe 仍然能达到 0.84-0.90 AUROC。
3. Gemma9B / Qwen3-4B 在 retrieved-doc gate 上强于 Gemma2B，MLP best 分别为 0.8995 / 0.8950。
4. TriviaQA AUROC 普遍高于 NQ，说明 retrieved-doc 是否有用在 TriviaQA 上更容易被 LLM hidden states 区分。
5. Step9R / Ours 需要超过这些静态 probe，才说明 CSV 注入在 pipeline-matched gate 上带来额外收益。
```

**Step9R / Ours 行后续填表条件**：Gemma2B 与 Qwen3-4B 已完整填写；Gemma2-9B 需要先跑 Step9R CSV_R。

---

### Step 8b：DPR Similarity Baseline [✓]

**做什么**：用 DPR encoder 直接算 `dot(q, d)`，**不依赖任何 LLM**。回答"我们的 LLM probe 是否真的比 DPR 检索分数更强"。

**命令**（已运行，结果保留作参考）：
```bash
python scripts/step8b_dpr_baseline.py --device cuda:1 --batch_size 64
```

**输出**：
```text
results/probe/dpr_baseline.json
```

**状态**：
- [✓] Encoder: facebook/dpr-{question,ctx}_encoder-single-nq-base
- [✓] 在 train + eval 上分别计算 (rel_score, dis_score)
- [✓] threshold 在 train 上校准, 应用到 eval

**结果（eval）**：

| Subset | AUROC_raw | AUROC_aligned | paired_acc | margin (rel-dis) | N |
|---|---:|---:|---:|---:|---:|
| NQ | 0.4022 ↓ | 0.5978 | 0.259 | -1.5156 | 648 |
| TriviaQA | 0.1748 ↓ | 0.8252 | 0.056 | -7.8221 | 824 |
| **Combined** | **0.2919 ↓** | **0.7081** | **0.145** | **-5.0459** | **1472** |

`↓` 表示 AUROC_raw < 0.5，即 DPR 分数与 relevance 标签 **反相关**：distract docs 平均分数 79.5162 > relevant docs 平均分数 74.4704。

---

#### 详细说明

**为什么 DPR 反相关？**

distracting docs 是 FAISS top-1 retrieval 的产物——**正是 DPR 自己认为"最相似"的文档**。而 relevant docs 是数据集 (NQ/TriviaQA) 自带的人类标注 gold passage。所以 DPR 系统性地给 distract 更高分。这不是 bug，而是数据集构造特性。

**TriviaQA 的反相关比 NQ 更强**（AUROC_raw 0.16 vs 0.41）：因为 DPR-NQ encoder 在 NQ 域内训练，但 TriviaQA 的 gold passage 风格与 DPR 训练数据差异更大，DPR 给 TriviaQA 的 gold 打分更低。

**为什么这是好消息**：

DPR baseline 几乎没有判别能力 (combined AUROC=0.29 反相关；aligned 后也只有 0.71)，这创造了与 LLM probe (MLP=0.83-0.86) 和 CSV (0.99+) 的强烈对比，是论文核心叙事的关键素材。

**Encoder 选择**：固定使用 `facebook/dpr-question_encoder-single-nq-base` + `facebook/dpr-ctx_encoder-single-nq-base`。NQ 是 DPR 的训练域，TriviaQA 是 out-of-domain，统一用同一对 encoder 是论文中典型的"DPR baseline"设置。

**评估方式**：每个样本拆成两个 (score, label) 对：
```
(dot(q, relevant_doc),   1)
(dot(q, distract_doc),   0)
```
所有 (score, label) 收齐后算 AUROC。Accuracy 用 train 校准的阈值应用到 eval。脚本同时报告 paired metrics（每对 q 的 rel vs dis 比较），更直观地反映 DPR 在每个样本上的判断方向。

---

#### 重要说明：raw AUROC / aligned AUROC / Acc=0.5 三个数怎么读

DPR baseline 的输出有三个看似矛盾的数字，实际上完整描述了它的行为：

```
raw AUROC      = 0.2919    score 与 label 反相关
aligned AUROC  = 0.7081    取了反向后的可分性
Accuracy       = 0.5000    任何 threshold 都给不出超过 majority 的判别
```

**1. raw AUROC = 0.29 < 0.5 不是 bug，是反相关**

AUROC 的定义是"随机抽一对 (positive, negative)，positive 分数高于 negative 的概率"。在我们的数据集上：

```
relevant docs (label=1) 平均 DPR dot = 74.4704
distract docs (label=0) 平均 DPR dot = 79.5162
```

distract 系统性地比 relevant 高约 5.05 — 因为 distract 是 FAISS top-1 retrieval 的产物，**正是 DPR 自己挑的"最相似"**。而 relevant 是数据集自带的人类标注 gold passage。所以"DPR 给 positive 打高分"这件事**几乎不发生**，AUROC 自然落到 0.5 以下。raw AUROC=0.29 真实地反映了"DPR 的 score 与 relevance label 反相关"。

**2. aligned AUROC = max(raw, 1-raw) = 0.71 = 反向后的可分性**

如果把 score 取个负号（即 "dis 高分=relevant" 反过来），AUROC 就变成 0.71。这告诉我们：DPR score 不是无信号，而是**信号方向错了**。`aligned AUROC` 的物理意义是"假设我们事先知道方向并取反，分类器还能达到 71% 左右的可分性"。

**3. Acc = 0.5000 = majority baseline**

`find_best_threshold` 在 train 上找最大化 Acc 的阈值。当 score 反相关时，任何 threshold 都得不到 > 0.5 的 Acc — 数学上找到的"最佳" threshold 会落在 score 分布最低值以下，导致**全部预测为 1**，Acc 退化为 n_pos/N = 0.5。**这不是阈值选择 bug**，而是反相关 score 的必然行为。

**论文里怎么报这个数**

| 数字 | 用途 | 备注 |
|---|---|---|
| **raw AUROC = 0.29** | 主表格主数字 | 最诚实，反映 DPR 实际表现；脚注说明反相关 |
| aligned AUROC = 0.71 | 主表格辅助列 | 显示"信号方向错了"——DPR 不是没有判别能力，而是判别方向与 relevance 相反 |
| Acc = 0.50 | 不单独报，或标 "N/A (anti-correlated)" | majority baseline，无信息量 |

**核心叙事**：DPR baseline 的反相关是一个**支持性发现**，而非反例。它表明：
- 我们的数据集 (distract = FAISS top-1) 在 DPR 空间里是**对抗性**的
- LLM hidden states (LR/MLP probe AUROC=0.83-0.86) 学到的是**与 DPR 正交**的判别信号，不是 retrieval-time similarity 的简单复读
- CSV (AUROC=0.99+) 进一步把这个正交信号放大到极致

---

### Step 9：CSV 训练

**做什么**：训练 Context Separator Vector，冻结 LLM 参数，只训练一个向量 v。在指定 transformer layer 注入 v，在指定 cls_layer 取 last-token hidden state，用 cosine similarity / temperature + centroid 做分类。

**数据输入口径**：Step9 直接读取 `data/final/train.json` 和 `data/final/eval.json` 构造 prompts，并在 forward 中注入 CSV 向量；它不直接使用 `data/hidden_states/{model}/*.pt`，因为那些是未注入 CSV 的静态 hidden-state 缓存，主要服务于 Step6b/7/8。

**两种模式**：
```text
Mode A：inject layer k, 最终层分类
Mode B：inject layer k, 指定 cls_layer c 分类, 要求 c > k
```

**入口脚本**：
```text
scripts/step9_train_csv.py              # 通用入口 (gemma2b + qwen3_4b 注册)
scripts/step9_train_csv_gemma9b.py      # gemma9b 专用 wrapper
scripts/step9_train_csv_qwen3_4b.py     # qwen3_4b 专用 wrapper
```

**Shell 脚本**（推荐用于大 sweep）：
```text
scripts/run_step9_dual_gpu.sh       # Gemma2-2B 旧 gold-vs-distractor CSV sweep，当前仓库实际存在
scripts/run_step9_gemma9b.sh        # Gemma2-9B 旧 CSV sweep
scripts/run_step9_qwen3_4b.sh       # Qwen3-4B 旧 CSV sweep
```

每个 shell 脚本支持 nohup 后台运行 + 自动 `--resume` + log 落盘 + PID 管理。

---

#### Step 9 — Gemma2-2B [✓]

**命令**（已完成）：

```bash
TARGET_MODEL=gemma2b \
GEMMA_GPU=0 \
GEMMA_BATCH_SIZE=8 \
INJECT_LAYERS=0,1,2,3,4,5 \
CLS_LAYERS=2,4,6,8,10,12,14,16,18,20,22,24,-1 \
NUM_EPOCHS=20 \
PATIENCE=5 \
bash scripts/run_step9_dual_gpu.sh
```

**输出**：
```text
results/csv/gemma2b_inject_{k}_cls_{c}_csv.pt
results/csv/gemma2b_inject_{k}_cls_{c}_csv_result.json
results/csv/gemma2b_csv_sweep.json
results/csv/gemma2b_csv_auroc_by_layer.png
```

**状态**：
- [✓] sweep 完成：6 个 inject × 13 个 cls = 78 个候选 pair；其中 `cls != -1` 且 `inject >= cls` 的组合会被跳过，实际有效 config 为 72 个
- [✓] best AUROC = 0.9964, best Acc = 0.9592 (inject=1, cls=16, epoch=16)
- [✓] 几乎所有 inject∈{0..4} × cls∈{4..24, -1} 配置均达到 AUROC ≥ 0.99
- [✓] 已用 step9b 重评 best ckpt 得到分 subset 数字 (NQ AUROC=0.9970, TriviaQA AUROC=0.9957)
- [✓] best ckpt 仅作为旧 gold-vs-distractor separability / CSV amplification 证据保留；不再作为 Step11 gate 输入

**Top 5 configs (按 best_auroc)**：

| Config | AUROC | Acc | Margin | Best Epoch |
|---|---:|---:|---:|---:|
| inject=1, cls=16 | **0.9964** | 0.9592 | 6.91 | 16 |
| inject=3, cls=14 | 0.9963 | 0.9606 | 6.84 | 18 |
| inject=0, cls=16 | 0.9963 | 0.9620 | 5.84 | 14 |
| inject=2, cls=14 | 0.9963 | 0.9626 | 6.92 | 10 |
| inject=4, cls=10 | 0.9963 | 0.9592 | 6.71 | 6 |

**最佳 config 选定**：`inject=1, cls=16, epoch=16`（sweep 中 best_auroc 最高 0.9964）。

注：Top 5 之间 AUROC 差距极小（< 0.0005）。上表按 `best_auroc` 排序；其中 `inject=0, cls=16` 的精确 AUROC=0.9963227，略高于 `inject=2, cls=14` 的 0.9963171。`inject=2, cls=14` 的 Acc 反而略高（0.9626），但这里仍以 AUROC 为主指标选 `inject=1, cls=16`。

**Pattern**：
- inject ∈ {0,1,2,3,4} 都 OK；inject=5 略下滑
- cls 在 4-24 层都很好；cls=2 不够用（信号未充分形成）；cls=-1（最终层）也不差但略逊于中层
- 最差 config: inject=1, cls=2 → 0.9670

---

#### Step 9 — Gemma2-9B（旧 gold-vs-distractor CSV，current sweep + Step9b 完成）[✓]

**历史命令 / 保留记录**（AUTODL / `/path/to/rag-probe`）：

```bash
INJECT_LAYERS=0,1,2,3,5 CLS_LAYERS=2,4,8,12,16,20,24,30,36,-1 bash scripts/run_step9_gemma9b.sh
```

**当前脚本参数口径**：
```text
GEMMA9B_GPU=0
GEMMA9B_BATCH_SIZE=12
NUM_EPOCHS=20
PATIENCE=5
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

原计划 5 × 10 = 50 个候选 pair；由于 Step9 会跳过 `cls != -1` 且 `inject >= cls` 的非法组合，原计划有效 config 为 46 个。当前本地汇总的 `gemma9b_csv_sweep.json` 含 **38 个 config**，因此在论文中应写作“current completed sweep / 当前记录”，除非后续补齐剩余 config。

**当前状态**：
- [✓] right-padding sanity check 已通过
- [✓] 当前汇总：`results/csv/gemma9b_csv_sweep.json`，共 38 个 config
- [✓] 当前 best：`inject=0, cls=30`, AUROC=0.997373, Acc=0.966033, epoch=6
- [✓] checkpoint 路径已修正到本地：`results/csv/gemma9b_inject_0_cls_30_csv.pt`
- [✓] Step9b per-subset 已完成，输出：`results/csv/gemma9b_csv_per_subset.json`

**Step9b per-subset 结果**：

| Subset | AUROC | Accuracy | Margin | N prompts |
|---|---:|---:|---:|---:|
| combined | **0.9974** | **0.9654** | 4.9616 | 1472 |
| NQ | 0.9983 | 0.9722 | 5.0805 | 648 |
| TriviaQA | 0.9962 | 0.9600 | 4.8682 | 824 |

**当前 best config**：

| Config | AUROC | Acc | Best Epoch | 备注 |
|---|---:|---:|---:|---|
| inject=0, cls=30 | **0.9974** | **0.9660** | 6 | 当前 38-config sweep best；Step9b 重评 combined AUROC 完全匹配 |

**本地 Step9b 命令**：

```bash
cd /path/to/rag-probe

export GEMMA9B_MODEL_PATH=google/gemma-2-9b
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 python scripts/step9b_reeval_per_subset.py   --model gemma9b   --batch_size 12
```

##### 详细说明：AUTODL 部署

无卡上传 gemma2 9B 模型，安装环境：

```bash
pip install -U \
  transformers accelerate safetensors huggingface_hub datasets \
  scikit-learn matplotlib tqdm numpy scipy pandas \
  "httpx[socks]" socksio nvitop
```

打包 4090 代码：

```bash
tar -czvf step9_gemma9b_bundle.tar.gz \
  scripts/step9_train_csv.py \
  scripts/step9_train_csv_gemma9b.py \
  scripts/run_step9_gemma9b.sh \
  csv_module/__init__.py \
  csv_module/llm_layers.py \
  csv_module/train_utils.py \
  data/final/train.json \
  data/final/eval.json
```

注意：Gemma2-9B 不建议在单张 4090 上正式大跑。如果日志出现 CPU offload 或 shard 加载特别慢，应停止并换 80GB GPU。

---

#### Step 9 — Qwen3-4B-Base（旧 gold-vs-distractor CSV，current sweep + Step9b 完成）[✓]

**历史命令 / 保留记录**（4090 单卡）：

```bash
QWEN3_4B_GPU=0 QWEN3_4B_BATCH_SIZE=4 INJECT_LAYERS=0,2,4,6,11 CLS_LAYERS=8,16,23,24,30,-1 NUM_EPOCHS=20 PATIENCE=5 LAM=5.0 LR=0.005 COS_TEMP=0.1 EMA_DECAY=0.999686 bash scripts/run_step9_qwen3_4b.sh
```

原计划 5 × 6 = 30 个候选 pair；其中 `inject=11, cls=8` 会被跳过，原计划有效 config 为 29 个。当前本地 `qwen3_4b_csv_sweep.json` 含 **24 个 config**，因此在论文中应写作“current completed sweep / 当前记录”，除非后续补齐剩余 config。

**当前状态**：
- [✓] `csv_module/llm_layers.py` 已支持 Qwen3DecoderLayer (Tensor 返回)
- [✓] 最小兼容测试通过 (inject=0, cls=2, batch=1, epoch=1)
- [✓] 当前汇总：`results/csv/qwen3_4b_csv_sweep.json`，共 24 个 config
- [✓] 当前 best：`inject=6, cls=24`, AUROC=0.996077, Acc=0.955163, epoch=9
- [✓] Step9b per-subset 已完成，输出：`results/csv/qwen3_4b_csv_per_subset.json`

**Step9b per-subset 结果**：

| Subset | AUROC | Accuracy | Margin | N prompts |
|---|---:|---:|---:|---:|
| combined | **0.9961** | **0.9552** | 5.2596 | 1472 |
| NQ | 0.9966 | 0.9583 | 5.4586 | 648 |
| TriviaQA | 0.9956 | 0.9527 | 5.1031 | 824 |

**当前 best config**：

| Config | AUROC | Acc | Best Epoch | 备注 |
|---|---:|---:|---:|---|
| inject=6, cls=24 | **0.9961** | **0.9552** | 9 | 当前 24-config sweep best；Step9b 重评 combined AUROC 完全匹配 |

**本地 Step9b 命令**：

```bash
cd /path/to/rag-probe

export QWEN3_4B_MODEL_PATH=Qwen/Qwen3-4B-Base
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 python scripts/step9b_reeval_per_subset.py   --model qwen3_4b   --batch_size 4
```

Qwen3-4B 的结果说明 CSV 跨架构仍然有效：Step8 MLP best 仅 0.8325，而旧 gold-vs-distractor CSV current sweep best 已达到 0.9961。

##### 详细说明：csv_module/llm_layers.py 的 Qwen3 适配

Qwen3DecoderLayer.forward 返回类型是 `torch.Tensor`，与 Gemma2/LLaMA 的 tuple 返回不同。`csv_module/llm_layers.py` 通过类名注册显式分支：

```python
_TENSOR_RETURNING_LAYERS = {
    "Qwen3DecoderLayer",
    "Qwen3MoeDecoderLayer",
}
```

wrapper 在 `__init__` 中检查 `type(decoder_layer).__name__` 是否在该集合内。命中即返回 Tensor，否则按 tuple 路径处理。Gemma2 (tuple) / LLaMA (tuple) / Qwen3 (Tensor) 三种返回类型在同一个 wrapper 中都能正确转发。

##### 显式参数测试（已完成的对照命令）

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/step9_train_csv.py \
  --model qwen3_4b \
  --str_layers 4,6,11 \
  --cls_layer -1 \
  --lam 5.0 \
  --lr 0.005 \
  --cos_temp 0.1 \
  --ema_decay 0.999686 \
  --batch_size 4 \
  --num_epochs 20 \
  --patience 20
```

##### 历史最小兼容测试（保留作记录）

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=0 \
TRANSFORMERS_OFFLINE=0 \
QWEN3_4B_MODEL_PATH=Qwen/Qwen3-4B-Base \
python scripts/step9_train_csv_qwen3_4b.py \
  --model qwen3_4b \
  --str_layer 0 --cls_layer 2 \
  --batch_size 1 --num_epochs 1 --patience 1

# 第一个有效配置: epoch 1 loss=0.5829, AUROC=0.9499, Acc=0.9164
python scripts/step9_train_csv_qwen3_4b.py --model qwen3_4b \
  --str_layer 0 --cls_layer 2 --num_epochs 20 --patience 5 --batch_size 4
```

注：Qwen3-4B 上目前观察到 cls=-1（最终层）有较好表现，与 Gemma2 的"cls 浅层是甜区"模式不同——这是值得在论文中讨论的跨架构差异。

---

#### Step 9 输出文件（统一）

每个模型统一保存：

```text
results/csv/{model}_inject_{k}_cls_{c}_csv.pt              # checkpoint
results/csv/{model}_inject_{k}_cls_{c}_csv_result.json     # 单 config 结果
results/csv/{model}_csv_sweep.json                         # sweep 汇总
results/csv/{model}_csv_auroc_by_layer.png                 # AUROC 地形图
```

checkpoint 内容（按当前 `step9_train_csv.py` 实际保存字段）：
```text
tsv, centroids, str_layer, cls_layer,
lam, cos_temp,
best_auroc, best_accuracy, best_avg_margin, best_epoch,
model_name, hf_name
```

注意：当前 checkpoint 不保存 `lr` 和 `ema_decay`。如果后续需要完整复现实验超参，建议在 Step9 保存 checkpoint 时补充这两个字段，或至少在 result json / sweep json 里保留。

---

#### Step 9 — 推荐搜索范围

Gemma2 系列标准 sweep（gemma2b 已用过）：
```text
inject layers = 0,1,2,3,4,5
cls layers    = 2,4,6,8,10,12,14,16,18,20,22,24,-1
```

Gemma2-9B AUTODL 当前 sweep（更稀疏，匹配 42 层模型）：
```text
inject layers = 0,1,2,3,5
cls layers    = 2,4,8,12,16,20,24,30,36,-1
```

Qwen3-4B 当前 sweep（基于已发现的较好组合）：
```text
inject layers = 0,2,4,6,11
cls layers    = 8,16,23,24,30,-1
ema_decay     = 0.999686    # 比默认 0.99 慢得多, 在 Qwen3 上效果更好
```

小网格测试（任意模型）：
```text
inject layers = 0,1,2
cls layers    = 2,4,6,8,12,-1
```

最小兼容性测试（新模型）：
```text
inject layer = 0
cls layer    = 2
batch_size   = 1
num_epochs   = 1
```

---

### Step 9R：Retrieved-doc CSV 训练（Step11 gate 主线）[~]

**做什么**：训练一个 pipeline-matched CSV scorer。与旧 Step9 不同，Step9R 不再比较 gold `relevant_doc` 与 `distracting_doc`，而是只看真实 pipeline 会拿到的 `retrieved_top1/doc`，预测这个 retrieved doc 是否 actually relevant。

**脚本**：
```text
scripts/step9_train_csv_retrieved.py
scripts/run_step9r_gemma2b.sh
scripts/run_step9r_qwen3_4b.sh
scripts/run_step9r_gemma9b.sh
```

**输入 / 输出**：
```text
输入：data/final_retrieved/train.json / eval.json
输出：results/csv_retrieved/{model}_retrieved_inject_{k}_cls_{c}_csv.pt
      results/csv_retrieved/{model}_retrieved_inject_{k}_cls_{c}_csv_result.json
      results/csv_retrieved/{model}_retrieved_csv_sweep.json
```

**固定随机种子**：`step9_train_csv_retrieved.py` 已固定 `FIXED_SEED=42`，并在每个 config 训练前重新 set seed。这样同一个 config 重新跑更稳定，且中途 `--resume` 不会因为前面 config 被跳过而改变后续 config 的随机状态。

#### Step9R — Gemma2-2B [✓]

**已完成 fixed-layer 调参**：固定 `inject=1, cls=16`，比较 lr / ema 后，当前最优为：

```text
lr=0.002
ema_decay=0.99
best AUROC≈0.8958
best epoch=6
```

**已完成大范围 expanded sweep**：

`scripts/run_step9r_gemma2b.sh` 已用于 Gemma2B retrieved-doc CSV 的 57-config expanded sweep。该脚本训练在：

```text
data/final_retrieved/{train,eval}.json
```

并保存到：

```text
results/csv_retrieved/
```

`scripts/run_step9r_gemma2b.sh` 默认参数：

```text
GEMMA_GPU=1
GEMMA_BATCH_SIZE=8
INJECT_LAYERS=1,2,3,4,5,6,10,12,16
CLS_LAYERS=8,12,16,18,20,24,-1
EXPECTED_CONFIGS=57
NUM_EPOCHS=20
PATIENCE=5
LAM=5.0
LR=0.002
COS_TEMP=0.1
EMA_DECAY=0.99
SEED=42（写死在 Python）
```

有效 config 数为 57：

```text
inject=1,2,3,4,5,6 each has 7 valid cls choices = 42
inject=10 has cls=12,16,18,20,24,-1 = 6
inject=12 has cls=16,18,20,24,-1 = 5
inject=16 has cls=18,20,24,-1 = 4
total = 57 valid configs
```

**已补跑固定 `CLS=18` inject sweep**：

在大范围 sweep 跑完后，又使用 `scripts/run_step9r_gemma2b_cls18_inject_sweep.sh` 固定 `cls=18`，补齐所有合法 inject layer：

```text
CLS_LAYERS=18
INJECT_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
EXPECTED_CONFIGS=18
LR=0.002
EMA_DECAY=0.99
```

这样做的原因是 Gemma2B retrieved-doc probe baseline 的最佳判别层在 `layer_18`，因此除了 expanded sweep 中已有的部分 `cls=18` 组合，还需要固定 `CLS=18` 细扫所有 `inject < 18` 的位置。

注意：如果该固定 `CLS=18` 脚本是在 57-config expanded sweep 之后运行，且没有 reset 旧结果，那么 `--resume` 会跳过已经完成的 `inject_1/2/3/4/5/6/10/12/16_cls_18` 等组合，实际新增的 unique config 主要是：

```text
inject=0,7,8,9,11,13,14,15,17 with cls=18
```

因此 Gemma2B Step9R 最终 best config 应该从同一个总 sweep 文件中选择：

```text
results/csv_retrieved/gemma2b_retrieved_csv_sweep.json
```

不要只看单独的 fixed-layer tuning 结果；也不要只看 broad sweep 的前 57 个 config。后续需要从该 sweep JSON 中按 `best_auroc` 选 best checkpoint，并建议再跑 retrieved-doc per-subset 重评。

**日志和结果保存**：

```text
# 57-config expanded sweep
logs/step9_retrieved/gemma2b_retrieved_sweep_inj1-6-10-12-16_cls8-12-16-18-20-24-final_lr0p002_ema0p99.log
logs/step9_retrieved/gemma2b_retrieved.pid

# fixed cls=18 inject sweep
logs/step9_retrieved/gemma2b_retrieved_cls18_inject_sweep_inj0-17_cls18_lr0p002_ema0p99.log
logs/step9_retrieved/gemma2b_retrieved_cls18_inject_sweep.pid

# shared outputs
results/csv_retrieved/gemma2b_retrieved_*.pt
results/csv_retrieved/gemma2b_retrieved_*_csv_result.json
results/csv_retrieved/gemma2b_retrieved_csv_sweep.json
```

**跑完后检查 best**：

```bash
python - <<'PYCODE'
import json
from pathlib import Path

p = Path("results/csv_retrieved/gemma2b_retrieved_csv_sweep.json")
sweep = json.load(open(p))
best_key = max(sweep, key=lambda k: sweep[k]["best_auroc"])
best = sweep[best_key]
print("configs:", len(sweep))
print("best_key:", best_key)
print("best_auroc:", best["best_auroc"])
print("best_accuracy:", best["best_accuracy"])
print("best_epoch:", best["best_epoch"])
print("checkpoint:", best["checkpoint"])
PYCODE
```

#### Step9R — Qwen3-4B [✓ / hparam grid 进行中]

**fixed-layer 调参结论**：固定 `inject=4, cls=30` 后，四组 `lr / ema_decay` 调参已经完成。当前以 `best AUROC` 为主指标，选择：

```text
best fixed-layer config: inject=4, cls=30
best hyperparams: lr=0.002, ema_decay=0.99
best AUROC≈0.8846
best Acc≈0.8244
best epoch=5
```

四组调参结论：

| lr | ema_decay | best AUROC | best Acc | best epoch | 结论 |
|---:|---:|---:|---:|---:|---|
| **0.002** | **0.99** | **≈0.8846** | ≈0.8244 | 5 | 主推荐 |
| 0.002 | 0.999686 | ≈0.8831 | **≈0.8267** | 7 | 很接近，Acc 略高 |
| 0.005 | 0.99 | ≈0.8293 | ≈0.7344 | 5 | 明显较差 |
| 0.005 | 0.999686 | ≈0.7907 | ≈0.7494 | 7 | 最差 |

解释：`lr=0.002` 明显优于 `lr=0.005`；`ema=0.99` 在 best AUROC 上略优于 `0.999686`，因此 expanded sweep 固定使用 `lr=0.002, ema_decay=0.99`。

**expanded layer sweep 已完成**：

```text
scripts/run_step9r_qwen3_4b.sh
```

该脚本训练在：

```text
data/final_retrieved/{train,eval}.json
```

并保存到：

```text
results/csv_retrieved/
```

expanded sweep 参数：

```text
QWEN3_4B_GPU=1
QWEN3_4B_BATCH_SIZE=4
INJECT_LAYERS=2,4,6,8,10,11
CLS_LAYERS=8,12,16,20,22,23,24,27,30,-1
EXPECTED_CONFIGS=57
NUM_EPOCHS=20
PATIENCE=5
LAM=5.0
LR=0.002
COS_TEMP=0.1
EMA_DECAY=0.99
SEED=42（写死在 Python）
```

有效 config 数：57。

```text
inject=2:  cls=8,12,16,20,22,23,24,27,30,-1 = 10
inject=4:  cls=8,12,16,20,22,23,24,27,30,-1 = 10
inject=6:  cls=8,12,16,20,22,23,24,27,30,-1 = 10
inject=8:  cls=8 is skipped, remaining 9
inject=10: cls=8 is skipped, remaining 9
inject=11: cls=8 is skipped, remaining 9
total = 57 valid configs
```

**为什么这版 Qwen sweep 比 v15 更大**：

Qwen3-4B retrieved-doc probe baseline 的 MLP best 是 `layer_23`，combined AUROC=0.8950；fixed-layer CSV 的 `cls=30` best AUROC≈0.8846，还没有超过该 baseline。因此 expanded sweep 不应只围绕 `cls=30`，而是需要覆盖：

```text
cls=23：覆盖 retrieved-doc probe baseline 最佳层
cls=24/30：覆盖已知中后层强候选与 fixed-layer tuning 位置
cls=22/27：补充 layer_23 附近和后层过渡区
cls=8/12/16/20：观察中层趋势
cls=-1：保留 final-layer 对照
```

**运行命令**：

如果 `results/csv_retrieved/` 中有旧 Qwen retrieved 结果，第一次用新超参跑建议备份旧结果：

```bash
cd /path/to/rag-probe
RESET_QWEN_RESULTS=1 bash scripts/run_step9r_qwen3_4b.sh
```

如果确认没有旧 Qwen retrieved 结果，或只想让 `--resume` 跳过已完成的新配置：

```bash
cd /path/to/rag-probe
bash scripts/run_step9r_qwen3_4b.sh
```

**日志和结果保存**：

```text
logs/step9_retrieved/qwen3_4b_retrieved_expanded_inj2-4-6-8-10-11_cls8-12-16-20-22-23-24-27-30-final_lr0p002_ema0p99.log
logs/step9_retrieved/qwen3_4b_retrieved.pid
results/csv_retrieved/qwen3_4b_retrieved_*.pt
results/csv_retrieved/qwen3_4b_retrieved_*_csv_result.json
results/csv_retrieved/qwen3_4b_retrieved_csv_sweep.json
```

**固定 `cls=22` inject sweep 已完成**：

由于 expanded sweep 与后续结果显示 `cls=22` 附近更强，已补跑固定 `cls=22`、`inject=0..21` 的完整 sweep：

```text
script: scripts/run_step9r_qwen3_4b_cls22_inject_sweep.sh
cls_layers: 22
inject_layers: 0..21
results: results/csv_retrieved/
```

固定 `cls=22` inject sweep 在 `results/csv_retrieved/` 下的最佳为：

```text
best config:  inject_19_cls_22
AUROC:        0.914968
Accuracy:     0.830445
best_epoch:   3
checkpoint:   results/csv_retrieved/qwen3_4b_retrieved_inject_19_cls_22_csv.pt
```

Top configs：

| config | AUROC | Acc | best epoch |
|---|---:|---:|---:|
| inject_19_cls_22 | 0.914968 | 0.830445 | 3 |
| inject_21_cls_22 | 0.913498 | 0.846838 | 15 |
| inject_20_cls_22 | 0.912935 | 0.800468 | 5 |
| inject_18_cls_22 | 0.912905 | 0.836534 | 3 |
| inject_10_cls_22 | 0.906128 | 0.823888 | 4 |

注意：`results/csv_retrieved/qwen3_4b_retrieved_csv_sweep.json` 当前主要记录固定 `cls=22` sweep 的 22 个 config；后续 hparam grid 已在独立目录运行，避免覆盖旧结果。

**cls22 hparam grid 已完成**：

```bash
bash scripts/run_step9r_qwen3_4b_cls22_hparam_grid.sh
```

该脚本按 LR/EMA 分目录保存结果，避免不同超参覆盖同名 Step9R 文件。当前最终选择用于 pipeline 的 Qwen3-4B CSV_R best 为：

```text
results_dir:   results/csv_retrieved_qwen3_4b_cls22_hparam_grid/lr0p001_ema0p999
best config:   inject_21_cls_22
AUROC:         0.915867
Accuracy:      0.855269
best_epoch:    20
checkpoint:    results/csv_retrieved_qwen3_4b_cls22_hparam_grid/lr0p001_ema0p999/qwen3_4b_retrieved_inject_21_cls_22_csv.pt
```

Top hparam-grid configs：

| run | config | AUROC | Acc | best epoch |
|---|---|---:|---:|---:|
| lr=0.001, ema=0.999 | inject_21_cls_22 | 0.915867 | 0.855269 | 20 |
| lr=0.001, ema=0.99 | inject_19_cls_22 | 0.915088 | 0.831850 | 3 |
| lr=0.002, ema=0.99 | inject_19_cls_22 | 0.914984 | 0.830445 | 3 |
| main sweep | inject_19_cls_22 | 0.914968 | 0.830445 | 3 |
| lr=0.0005, ema=0.999 | inject_21_cls_22 | 0.913981 | 0.858548 | 20 |

为避免 Step11 默认读取 `results/csv_retrieved/` 中的次优 main sweep，已整理 pipeline 专用目录：

```text
results/csv_retrieved_qwen3_4b_best_for_pipeline/
  qwen3_4b_retrieved_csv_sweep.json
  qwen3_4b_retrieved_inject_21_cls_22_csv.pt
  qwen3_4b_retrieved_inject_21_cls_22_csv_result.json
```

运行 Qwen3-4B pipeline 时需设置：

```bash
export CSV_RETRIEVED_DIR=/path/to/rag-probe/results/csv_retrieved_qwen3_4b_best_for_pipeline
```

#### Step9R — Gemma2-9B [ ]

Gemma2-9B retrieved-doc sweep 暂时不优先跑。计划：等 Gemma2B / Qwen3-4B 的 Step9R gate 和 Step11 baseline 路线稳定后，把以下内容迁移到 AUTODL：

```text
data/final_retrieved/
scripts/step9_train_csv_retrieved.py
scripts/run_step9r_gemma9b.sh
csv_module/
```

默认可用配置暂定：
```text
INJECT_LAYERS=0,1,2,3,5
CLS_LAYERS=2,4,8,12,16,20,24,30,36,-1
batch_size=12
```

---

### 第二阶段总结表 [✓]

> **关于 DPR Similarity baseline**：DPR Sim 不依赖 LLM，单独在 §Step 8b 报告（combined AUROC_raw=0.29 反相关，aligned=0.71；combined Acc=0.50 majority baseline），故不纳入此分类器对比表。

#### 分类器对比表（best layer 上分 subset）

| 模型 | Classifier | combined AUROC | NQ AUROC | TriviaQA AUROC | combined Acc | NQ Acc | TriviaQA Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma2-2B  | LR       | 0.8029 | 0.7703 | 0.7531 | 0.7446 | 0.7114 | 0.7039 |
| Gemma2-2B  | MLP      | 0.8568 | 0.8124 | 0.8401 | 0.7779 | 0.7407 | 0.7743 |
| Gemma2-2B  | Centroid | 0.7458 | 0.7350 | 0.7175 | 0.6855 | 0.6651 | 0.6638 |
| **Gemma2-2B** | **CSV**     | **0.9964** | **0.9970** | **0.9957** | **0.9592** | **0.9614** | **0.9575** |
| Gemma2-9B  | LR       | 0.7605 | 0.6849 | 0.8241 | 0.7052 | 0.6343 | 0.7524 |
| Gemma2-9B  | MLP      | 0.7702 | 0.6723 | 0.8089 | 0.7052 | 0.6250 | 0.7415 |
| Gemma2-9B  | Centroid | 0.7415 | 0.6515 | 0.7964 | 0.6712 | 0.5988 | 0.7148 |
| **Gemma2-9B** | **CSV**     | **0.9974** | **0.9983** | **0.9962** | **0.9654** | **0.9722** | **0.9600** |
| Qwen3-4B   | LR       | 0.8127 | 0.7729 | 0.8391 | 0.7486 | 0.6806 | 0.7779 |
| Qwen3-4B   | MLP      | 0.8325 | 0.7357 | 0.8483 | 0.7636 | 0.6698 | 0.7803 |
| Qwen3-4B   | Centroid | 0.7768 | 0.7527 | 0.8126 | 0.7154 | 0.6883 | 0.7439 |
| **Qwen3-4B**  | **CSV**     | **0.9961** | **0.9966** | **0.9956** | **0.9552** | **0.9583** | **0.9527** |

**填表口径说明**：

- **LR / MLP / Centroid / Mass-Mean 行**：每个 (模型, classifier) 组合，先用 combined AUROC 选出 best layer，然后**这一层**对应的所有 6 个数字一起填。NQ/TriviaQA 列是 best layer 上的数字，**不是各 subset 自己的最佳**。
  - Gemma2-2B 四种 classifier 的 best layer 都是 layer 5
  - Gemma2-9B：LR / Centroid / Mass-Mean 在 layer 29，MLP 在 layer 31
  - Qwen3-4B 四种 classifier 的 best layer 都是 layer 23
- **CSV 行**：填 `step9b_reeval_per_subset.py` 加载 best CSV checkpoint 后分 subset重新评估的结果。三模型 CSV 行现在均已填实。
- Gemma2-2B best ckpt: inject=1 / cls=16；Gemma2-9B best ckpt: inject=0 / cls=30；Qwen3-4B best ckpt: inject=6 / cls=24。

#### 如何填 CSV 行的数字

```bash
# Gemma2-2B (sweep 已完成,可立即跑)
python scripts/step9b_reeval_per_subset.py --model gemma2b

# 其余等 sweep 完成后再跑
python scripts/step9b_reeval_per_subset.py --model gemma9b
python scripts/step9b_reeval_per_subset.py --model qwen3_4b

# 或一次跑全部已就绪的
python scripts/step9b_reeval_per_subset.py --model all
```

输出在 `results/csv/{model}_csv_per_subset.json`，把里面 `results.combined / results.nq / results.triviaqa` 的 AUROC 和 Accuracy 填到上表对应的 CSV 行。脚本会自动校验：重新计算的 combined AUROC 应与 sweep 里保存的 best_auroc 在 0.001 以内一致。

---

#### 观察

- **Gemma2-9B / Qwen3-4B 上 TriviaQA AUROC > NQ AUROC**：与 DPR baseline 在 TriviaQA 上反相关更强（DPR encoder 在 TriviaQA 域外）一致 —— LLM hidden states 在 TriviaQA 上反而更可分。
- **Gemma2-2B 例外**：NQ 与 TriviaQA AUROC 接近，可能与 2B 模型容量有限、representation 的 domain bias 较弱有关。
- **MLP > LR > Centroid 在三个模型上一致**——非线性 probe 始终带来 0.02~0.05 AUROC 提升。

---

#### 最终 classifier 选择优先级

```text
1. AUROC
2. Accuracy
3. Margin / score gap (可分性裕度)
4. 跨 NQ / TriviaQA 稳定性
5. 是否容易接入 Step 11 classifier plugin / gate (CSV 直接给 v + centroid, 最直接)
```

三模型旧 CSV 均显著优于各自 MLP probe：
- Gemma2-2B: CSV 0.9964 vs MLP 0.8568
- Gemma2-9B: CSV 0.9974 vs MLP 0.7702
- Qwen3-4B: CSV 0.9961 vs MLP 0.8325

旧 CSV 行用于证明 hidden-state separability / CSV amplification；Step11 gate 主线仍使用新 Step9R retrieved-doc CSV。

---

## 五、第三阶段：Logit Lens 与 classifier-gated pipeline

---

### Step 10：Logit Lens 分析 [✓]

**做什么**：在 `data/final/eval.json` 的完整 736 条 eval samples 上，用 teacher forcing + logit lens 分析模型在不同层是否产生正确答案信号，并验证 relevant document 是否真的提高正确答案 token 的 log probability。

**核心问题**：
```text
Q1: gold_ctx 是否系统性提高 logP(answer)？
Q2: gold_ctx - no_ctx 的增益在哪一层最大？
Q3: distracting_ctx 是否不会提高、甚至降低正确答案信号？
```

**已运行命令**：
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/step10_logit_lens.py --model gemma2b
CUDA_VISIBLE_DEVICES=1 python scripts/step10_logit_lens.py --model gemma9b
CUDA_VISIBLE_DEVICES=1 python scripts/step10_logit_lens.py --model qwen3_4b
```

**输出**：
```text
results/logit_lens/gemma2b_logit_lens.json
results/logit_lens/gemma9b_logit_lens.json
results/logit_lens/qwen3_4b_logit_lens.json
```

Step10b 画图输出：
```text
results/logit_lens/gemma2b_logp_by_layer.png
results/logit_lens/gemma2b_delta_by_layer.png
results/logit_lens/gemma2b_logp_by_subset.png
results/logit_lens/gemma9b_logp_by_layer.png
results/logit_lens/gemma9b_delta_by_layer.png
results/logit_lens/gemma9b_logp_by_subset.png
results/logit_lens/qwen3_4b_logp_by_layer.png
results/logit_lens/qwen3_4b_delta_by_layer.png
results/logit_lens/qwen3_4b_logp_by_subset.png
```

说明：Step10b 只读取 Step10 生成的 json 并保存图片，不重新加载模型；当前三模型图片均已成功生成。

**状态**：
- [✓] step10_logit_lens.py 实现完成
- [✓] Gemma2-2B 完整 eval 分析完成
- [✓] Gemma2-9B 完整 eval 分析完成
- [✓] Qwen3-4B 完整 eval 分析完成
- [✓] Gemma2 final logit softcapping 处理完成
- [✓] final-layer double norm bug 已修正并通过 sanity check
- [✓] step10b_plot_logit_lens.py 画图完成：三模型 logP 曲线、ΔlogP 曲线、按 subset 曲线均已生成

---

#### Step 10 实现细节

三种分析条件：
```text
no_ctx      = Question: q
Answer:
gold_ctx    = Document: relevant_doc

Question: q
Answer:
dis_ctx     = Document: distracting_doc

Question: q
Answer:
```

原 Step10 分析 `gold_ctx / dis_ctx / no_ctx`，用于证明 gold evidence 会增强答案信号。由于 Step11 真实输入是 retrieved top-1 文档，当前已新增 Step10R retrieved-doc 版本，专门分析 `retrieved_label=1 / retrieved_label=0 / no_ctx`。

核心指标：
```text
mean_logP(answer | no_ctx)       # teacher forcing 下答案 token 平均 logP
mean_logP(answer | gold_ctx)
mean_logP(answer | dis_ctx)
Δgold-no = logP(answer | gold_ctx) - logP(answer | no_ctx)
Δdis-no  = logP(answer | dis_ctx)  - logP(answer | no_ctx)
delta_gold_peak_layer            # Δgold-no 最大层
answer_signal_emerge_layer       # gold_ctx 下 mean_logP 首次超过 -5 的层
```

多答案处理：
```text
1. 对每个 sample 的多个 gold answer aliases，先在 no_ctx 下计算每个 alias 的 mean-over-layers logP。
2. 选择 no_ctx 下模型最自然的 alias 作为 canonical answer。
3. no_ctx / gold_ctx / dis_ctx 三种条件都使用同一个 canonical answer。
```

这样可以避免用 gold_ctx 选择答案表述导致的偏置，同时避免只使用 `answers[0]` 低估模型能力。

Gemma2 特殊处理：
```python
# 中间层需要 final_norm 做 logit lens
# 最后一层 hidden_states[-1] 已经是 final norm 后的结果，不能再 norm 一次
if layer_idx == num_hidden_states - 1:
    hs_for_lm = hs
else:
    hs_for_lm = final_norm(hs)

logits = lm_head(hs_for_lm)

if softcap is not None:
    logits = softcap * torch.tanh(logits / softcap)
```

三模型 final-layer sanity check 均通过：
```text
Gemma2-2B: max abs diff = 0.00000000, argmax match = True
Gemma2-9B: max abs diff = 0.00000000, argmax match = True
Qwen3-4B:  max abs diff = 0.00000000, argmax match = True
```

---

#### Step 10 完整结果

| 模型 | hidden states 数 | final no_ctx logP | final no_ctx P | final gold_ctx logP | final gold_ctx P | final dis_ctx logP | final dis_ctx P | Δgold-no peak layer | peak Δgold-no | final Δdis-no | emerge layer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma2-2B | 27 | -3.0843 | 0.0458 | -2.1769 | 0.1134 | -3.1075 | 0.0447 | 23 | +2.4700 | -0.0232 | 23 |
| Gemma2-9B | 43 | -2.6830 | 0.0684 | -2.0609 | 0.1273 | -2.6662 | 0.0695 | 36 | +1.5295 | +0.0168 | 37 |
| Qwen3-4B | 37 | -2.9815 | 0.0507 | -1.8313 | 0.1602 | -3.1000 | 0.0450 | 31 | +3.4918 | -0.1185 | 32 |

`P≈exp(mean_logP)`，表示答案 token 的平均概率强度，不是准确率 / EM / F1。

---

#### Step 10 观察

**1. 三个模型均满足 gold_ctx > no_ctx。**

```text
Gemma2-2B: 0.1134 / 0.0458 ≈ 2.48×
Gemma2-9B: 0.1273 / 0.0684 ≈ 1.86×
Qwen3-4B:  0.1602 / 0.0507 ≈ 3.16×
```

这说明 relevant document 会稳定增强模型对正确答案的概率信号。

**2. distracting document 不产生类似增益。**

```text
Gemma2-2B: dis_ctx P=0.0447 < no_ctx P=0.0458
Gemma2-9B: dis_ctx P=0.0695 ≈ no_ctx P=0.0684
Qwen3-4B:  dis_ctx P=0.0450 < no_ctx P=0.0507
```

Gemma2-9B 的 final Δdis-no 为 +0.0168，但幅度非常小，基本可解释为 dis_ctx 与 no_ctx 持平；Qwen3-4B 与 Gemma2-2B 则表现为 dis_ctx 略微压低正确答案信号。

**3. gold document 的最大增益稳定出现在后层。**

```text
Gemma2-2B: peak layer 23 / 26 ≈ 88%
Gemma2-9B: peak layer 36 / 42 ≈ 86%
Qwen3-4B:  peak layer 31 / 36 ≈ 86%
```

这说明不同架构、不同规模模型中，gold context 带来的答案信号都主要在后 10%-15% 的层形成。当前不把这些层作为新的主解码方法线，而是作为解释材料：当 classifier 判断 retrieved document 可信时，模型内部确实存在更强答案信号；当 retrieved document 不可信时，应回退或降低现有 RAG/decoding 方法的影响。

```text
Gemma2-2B: relevant answer signal peak around layer 23
Gemma2-9B: relevant answer signal peak around layer 36
Qwen3-4B:  relevant answer signal peak around layer 31
```

**4. Qwen3-4B 是 Step10 中 gold-context 增益最强的模型。**

Qwen3-4B 的 final gold/no ratio = 3.16×，peak Δgold-no = +3.4918，均高于两个 Gemma2 模型。它对 relevant document 最敏感，且 distracting document 明确低于 no_ctx，因此后续尤其值得测试 classifier plugin 能否在 Qwen pipeline 上稳定提升已有 baseline。

**5. Gemma2-9B 的 no-context 先验更强。**

Gemma2-9B 的 no_ctx P=0.0684，高于 Gemma2-2B 和 Qwen3-4B，说明 9B 模型不依赖文档时也能给正确答案更高概率。但也因此它从 gold document 获得的相对增益较小。

---

#### Step 10 对论文叙事的意义

Step 10 为 Step 11 的 classifier plugin 提供了直接证据：

```text
有用文档 relevant_doc 会显著提高答案 logP；
distracting_doc 不会提高答案 logP；
因此已有 RAG / decoding 方法应该只在文档质量信号较强时启用，或由 classifier confidence 控制强度。
```

这与 Step 8/9 的文档质量分类结果形成互补：

```text
Step 8/9: hidden states 能判断文档是否 useful
Step 10: useful 文档确实会在后层增强答案概率信号
Step 11: 用 retrieved-doc classifier 作为 plugin/gate，控制是否启用已有 RAG 或 decoding baseline
```

---

### Step 10R：Retrieved-doc Logit Lens [✓/部分完成]

**做什么**：在 `data/final_retrieved/eval.json` 上，用真实 retrieved top-1 文档重新做 logit lens。该分析用于确认 gold-doc Step10 找到的后层 peak 是否也适用于真实 pipeline retrieved docs。

**已新增脚本**：
```text
scripts/step10_logit_lens_retrieved.py
```

**运行命令（Gemma2B/NQ 已完成）**：
```bash
python \
  scripts/step10_logit_lens_retrieved.py \
  --model gemma2b \
  --dataset nq \
  --device cuda:0
```

**输出文件**：
```text
results/logit_lens_retrieved/gemma2b_retrieved_logit_lens_nq.json
```

**Gemma2B/NQ 关键结果**：
```text
samples:    1152
relevant:   839
irrelevant: 313

retrieved relevant peak layer = 23
peak Δ(relevant-no) = +4.1748
answer signal emerge layer = 23
```

Top relevant layers：
```text
layer 23: relevant-no +4.1748, irrelevant-no +0.2521
layer 21: relevant-no +3.7865, irrelevant-no +0.0692
layer 20: relevant-no +3.6759, irrelevant-no +0.0622
layer 24: relevant-no +3.5801, irrelevant-no +0.2960
layer 25: relevant-no +2.8583, irrelevant-no +0.4347
layer 22: relevant-no +2.4805, irrelevant-no -0.2178
layer 19: relevant-no +2.0392, irrelevant-no -0.0903
layer 26: relevant-no +1.5913, irrelevant-no +0.0381
```

结论：retrieved relevant 的最佳层仍然是 layer 23，与 gold-doc Step10 一致。因此后续新解码方法优先使用 `contrast_layer=23`，备选层为 `21 / 20 / 24`。

---

### Step 11：Classifier-gated Pipeline Generation [✓]

**做什么**：在 `data/final/test_retrieval_nq.json` 和 `test_retrieval_triviaqa.json` 上生成答案。当前 Step11 同时支持 6 个 external / non-gated baseline，以及把 Step9R retrieved-doc classifier 作为 plugin/gate 接入已有方法。Gemma2B 的 Step9R checkpoint 已用于本地完整 pipeline；Qwen3-4B 的 CSV_R best 已用于 5090 pipeline，并已回传到 `autodl_5090_2/results/pipeline/`；Gemma2-9B gated 方法暂缓。

**当前代码状态**：

```text
[✓] step11_contrastive_decoding.py 已更新为 greedy-only 主实验脚本
[✓] baseline 方法不加载 CSV checkpoint，不做 CSV validation，不 score test docs
[✓] gated 方法默认读取 results/csv_retrieved/{model}_retrieved_csv_sweep.json；也可用 CSV_RETRIEVED_DIR 指定模型专用 best 目录
[✓] 三模型默认使用本地 snapshot 路径，避免重复访问 HuggingFace Hub
[✓] DoLA 使用 transformers-community/dola 的 custom generate；当前采用本地 generate.py，避免运行时访问 hf-mirror
[✓] 6 个 baseline 的 Gemma2B / NQ limit=2 smoke test 已通过
[✓] `context_ucd` 已从旧 probability-diff 版本改为 Context-UCD-Energy：使用 doc/no-doc 两路 logits 的 partition energy 权重
[✓] Gemma2B × 2 datasets × 6 baseline methods 已有正式 Step12 结果
[✓] `scripts/run_step11_cad_alpha_sweep_with_eval.sh` 已支持 `METHOD / ALPHAS / TAUS / CONTRAST_LAYERS`
[✓] `step11_contrastive_decoding.py` 已新增多个 classifier-plugin 版本：vanilla RAG / CAD / ACD / Context-UCD / DoLA 均可被 gate 控制
[✓] Gemma2B pipeline 已按固定 `alpha=0.5` 在 5090 服务器完成 5 个 plugin 对照，并补齐 `tau=0.5/0.4/0.3`
[✓] Qwen3-4B pipeline 已在 5090 服务器按同口径完成，并已回传到 `autodl_5090_2/results/pipeline/`
```

**Gemma2B score-only 检查状态（不生成答案）**：

当前已新增/使用 `scripts/step11_score_test_docs_only.py` 对最终 pipeline test retrieved docs 做分类器打分检查。该脚本流程为：

```text
1. 从 results/csv_retrieved/gemma2b_retrieved_csv_sweep.json 选择 best_auroc 最高的 checkpoint
2. 加载 Gemma2B base model 和 tokenizer
3. 检查 checkpoint metadata
4. 注入 CSV vector
5. 先在 data/final_retrieved/eval.json 上重新验证 AUROC / Acc
6. 再对 data/final/test_retrieval_{nq,triviaqa}.json 的 retrieved_doc 计算 P(relevant)
```

注意：最终测试集 `test_retrieval_{nq,triviaqa}.json` 没有 relevant/distracting label，因此 score-only 只能分析 `P(relevant)` 分布、`trusted@tau` 比例和人工抽样，不计算 test classification AUROC / Accuracy。

---

#### Step11 当前方法列表

**Baseline / external decoding methods（不使用 CSV）**：

```text
no_doc
vanilla_rag
cad_fixed
acd
context_ucd
dola
```

**Gated methods（Gemma2B 已有 Step9R retrieved-doc CSV；Qwen3-4B/Gemma9B 等待各自 best）**：

```text
csv_gated_vanilla_rag
csv_gated_cad
csv_gated_acd
csv_gated_context_ucd
csv_gated_dola
```

说明：`context_ucd` 不是原始 UCD。原始 UCD 是 expert-vs-amateur model contrast，需要同 tokenizer 的 smaller amateur model；当前为了三模型统一比较，采用的是 **Context-UCD-Energy / UCD-style context contrast**：同一个模型下对比 `Document + Question` 与 `Question only` 两路 logits，并用 UCD-style partition energy 计算权重。

---

#### Greedy-only 设置

当前 Step11 主实验统一使用 greedy decoding：

```text
temperature = 0.0
top_p = 1.0
HF DoLA: do_sample=False
自写 CAD / ACD / Context-UCD loop: argmax next token
```

脚本会拒绝 `temperature > 0`，避免 sampling 分支带来额外变量。

---

#### 各 baseline 的解码定义

| 方法 | 输入 / 对比对象 | 解码空间 | CSV 依赖 | 说明 |
|---|---|---|---:|---|
| `no_doc` | `Question only` | standard greedy | 否 | 测 parametric knowledge |
| `vanilla_rag` | `Document + Question` | standard greedy | 否 | 基础 RAG baseline |
| `cad_fixed` | `doc_prompt` vs `no_doc_prompt` | logits | 否 | 固定 α CAD |
| `acd` | `doc_prompt` vs `no_doc_prompt` | logits + entropy adaptive | 否 | entropy ratio 动态融合 |
| `context_ucd` | `doc_prompt` vs `no_doc_prompt` | energy-weighted logits | 否 | UCD-style energy-weighted context contrast；不是原始 UCD |
| `dola` | same `doc_prompt`, final layer vs earlier layers | HF DoLA custom generate | 否 | 使用官方 `transformers-community/dola` 实现 |

**CAD 公式**：

```text
logits = (1 + α) * logits(Document + Question) - α * logits(Question)
```

**ACD 公式**：

```text
d_alpha = H(no_ctx) / (H(doc_ctx) + H(no_ctx))
logits = logits_no + alpha * d_alpha * (logits_doc - logits_no)
```

**Context-UCD-Energy 公式**：

当前 `context_ucd` 不是旧版 probability-diff 写法，而是 UCD-style energy-weighted context contrast。两路输入为：

```text
base_logits    = logits(Document + Question)
amateur_logits = logits(Question only)
```

每一步用两路 path-history token logit 更新能量：

```text
E_doc = T * logsumexp((base_logits + logit_doc_prev) / T)
E_no  = T * logsumexp((amateur_logits + logit_no_prev) / T)
w_doc = E_doc / (E_doc + E_no)
w_no  = E_no  / (E_doc + E_no)
score = 2 * w_doc * base_logits - w_no * amateur_logits
next_token = argmax(score)
```

主实验固定：

```text
context_ucd_beta = 1.0
context_ucd_relative_top = 0.0
energy_temperature = 1.0
temperature = 0.0
```

实现细节：`Context-UCD-Energy` 使用 doc/no-doc 两套独立 KV cache；sampling 和 `relative_top` 在主实验中禁用，脚本会要求 `--temperature 0.0` 与 `--context_ucd_relative_top 0.0`。

---

#### 本地模型路径设置

当前 Step11 的 `MODEL_REGISTRY` 已改为本地 snapshot 地址优先：

```text
gemma2b:
  google/gemma-2-2b

gemma9b:
  google/gemma-2-9b

qwen3_4b:
  Qwen/Qwen3-4B-Base
```

仍保留环境变量覆盖：

```text
GEMMA_MODEL_PATH
GEMMA9B_MODEL_PATH
QWEN3_4B_MODEL_PATH
```

如果某台机器本地路径不同，可在命令前临时覆盖。

---

#### DoLA 本地 custom generate

当前 Transformers 会把 DoLA 移到 `transformers-community/dola` custom generation。为了避免每次运行访问 `hf-mirror.com`，当前采用本地文件：

```text
third_party/transformers-community-dola/custom_generate/generate.py
```

如果该文件不存在，可从 HuggingFace cache 复制：

```bash
mkdir -p third_party/transformers-community-dola/custom_generate

cp /path/to/transformers-community-dola/custom_generate/generate.py \
   third_party/transformers-community-dola/custom_generate/generate.py
```

运行时通过环境变量指定：

```bash
DOLA_CUSTOM_GENERATE=$PWD/third_party/transformers-community-dola/custom_generate/generate.py
```

---

#### 当前已通过的 smoke test

Gemma2B / NQ / `limit=2` 已确认以下 6 个 baseline 均能生成并保存输出：

```text
no_doc        ✓
vanilla_rag   ✓
cad_fixed     ✓
acd           ✓
context_ucd   ✓
dola          ✓
```

对应输出示例：

```text
results/pipeline/gemma2b_nq_no_doc_alpha1p0_tau0p5_temp0p0_limit2.json
results/pipeline/gemma2b_nq_vanilla_rag_alpha1p0_tau0p5_temp0p0_limit2.json
results/pipeline/gemma2b_nq_cad_fixed_alpha1p0_tau0p5_temp0p0_limit2.json
results/pipeline/gemma2b_nq_acd_alpha1p0_tau0p5_temp0p0_limit2.json
results/pipeline/gemma2b_nq_context_ucd_alpha1p0_tau0p5_temp0p0_limit2.json
results/pipeline/gemma2b_nq_dola_alpha1p0_tau0p5_temp0p0_limit2.json
```

说明：DoLA 可能仍打印 `output_hidden_states` generation flag warning，但当前流程已能完成并保存结果；只要输出 JSON 生成成功，warning 不影响主流程。

---

#### 当前正式运行命令与结果位置

Baseline 已完成，运行口径如下：

```bash
for MODEL in gemma2b qwen3_4b gemma9b; do
  DOLA_CUSTOM_GENERATE=$PWD/third_party/transformers-community-dola/custom_generate/generate.py \
  SKIP_EXISTING=1 \
  bash scripts/run_step11_greedy_baselines_with_eval.sh "$MODEL" all 0
done
```

该命令展开为：

```text
models:   gemma2b, qwen3_4b, gemma9b
datasets: nq, triviaqa
methods:  no_doc, vanilla_rag, cad_fixed, acd, context_ucd, dola
GPU:      CUDA_VISIBLE_DEVICES=0
```

历史全量 baseline 曾按三模型命令规划；当前本地正式 `results/pipeline/` 保留 Gemma2B 5090 最新版，Qwen3-4B pipeline 已在 `autodl_5090_2/results/pipeline/` 完成，Gemma2-9B pipeline 暂缓。

```text
Gemma2B: 2 datasets × (6 baseline + 5 plugin × 3 tau) = 42 summary rows
Qwen3-4B: 2 datasets × (6 baseline + 5 plugin × 3 tau) = 42 summary rows，已回传到 `autodl_5090_2/results/pipeline/`
```

结果位置：
```text
generation: results/pipeline/{model}_{dataset}_{method}_alpha...json
eval:       results/pipeline/eval_{model}_{dataset}_{method}_alpha...json
summary:    results/pipeline/pipeline_eval_summary.csv
```

Gemma2B 服务器重跑命令（固定超参主实验口径）：
```bash
cd /path/to/rag-probe

export GEMMA_MODEL_PATH=google/gemma-2-2b
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHON_BIN=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAX_INPUT_LENGTH=512 CAD_ALPHA=0.5 SKIP_EXISTING=1 \
  bash scripts/run_step11_greedy_baselines_with_eval.sh gemma2b all 0

for METHOD in csv_gated_vanilla_rag csv_gated_cad csv_gated_acd csv_gated_context_ucd csv_gated_dola; do
  METHOD="$METHOD" \
  ALPHAS="0.5" \
  TAU=0.5 \
  MAX_INPUT_LENGTH=512 \
  SKIP_EXISTING=1 \
    bash scripts/run_step11_cad_alpha_sweep_with_eval.sh gemma2b all 0
done
```

目的：比较每个原 baseline 与 “baseline + retrieved-doc classifier gate”。若 `P(relevant) >= 0.5`，运行对应 baseline；否则退回 `no_doc`。结果保存到 `results/pipeline/`，统一汇总到 `results/pipeline/pipeline_eval_summary.csv` 和 `results/all_finished_experiments.csv`。

Gemma2B `tau=0.4/0.3` 补跑计划：

```text
代码：无需修改；继续使用 scripts/run_step11_cad_alpha_sweep_with_eval.sh
任务：5 methods × 2 datasets × 2 tau = 20 generation jobs
推荐运行：5090 单卡上用 4-worker 队列限制最多 4 个并发
关键设置：RUN_EVAL=0 并行生成，全部结束后统一运行 Step12/collect
日志：logs/step11_tau03_04_parallel/
输出：results/pipeline/gemma2b_{dataset}_{method}_alpha0p5_tau{0p4,0p3}_temp0p0.json
```

---

#### 输出路径

Step11 generation JSON：

```text
results/pipeline/{model}_{dataset}_{method}_alpha{alpha}_tau{tau}_temp{temperature}.json
```

Step12 eval JSON：

```text
results/pipeline/eval_{model}_{dataset}_{method}_alpha{alpha}_tau{tau}_temp{temperature}.json
```

全局汇总：

```text
results/pipeline/pipeline_eval_summary.csv
results/pipeline/pipeline_eval_summary.json
```

---

#### Gated 方法单文件备用命令

如果需要单独跑某个 gated generation 文件，可直接调用 Step11：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/step11_contrastive_decoding.py \
  --model gemma2b \
  --dataset nq \
  --method csv_gated_cad \
  --alpha 0.5 \
  --tau 0.5 \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 20
```

CSV gate 逻辑：

```text
csv_filter:
  if P(relevant) >= tau: vanilla_rag
  else: no_doc

csv_gated_vanilla_rag:
  if P(relevant) >= tau: vanilla_rag
  else: no_doc

csv_gated_cad:
  if P(relevant) >= tau: CAD
  else: no_doc

csv_gated_acd:
  if P(relevant) >= tau: ACD
  else: no_doc

csv_gated_context_ucd:
  if P(relevant) >= tau: Context-UCD
  else: no_doc

csv_gated_dola:
  if P(relevant) >= tau: DoLA
  else: no_doc
```

---

## 六、第四阶段：端到端 Pipeline 评估

---

### Step 12：Pipeline EM / F1 评估 [~]

**做什么**：读取 Step11 generation JSON，计算 NQ / TriviaQA 的 Exact Match (EM) 和 token-level F1。

**重要更新**：Step12 已修复 answers 字符串 list 解析问题。之前如果 gold answers 保存为字符串形式：

```text
"['Scorpio', 'Skorpio', 'Scorpio (disambiguation)']"
```

旧逻辑会把它当作一个完整 gold answer，导致 `prediction="Scorpio"` 的 EM 被低估为 0。修复后会解析为多个 alias，并对 aliases 取 max EM/F1。

---

#### 当前评测方式

现在 baseline 批量脚本已经把 Step12 接在 Step11 后面：

```text
scripts/run_step11_greedy_baselines_with_eval.sh
```

因此正式跑 baseline 时，不需要手动逐个调用 Step12。脚本逻辑是：

```text
1. 对某个 model + dataset 依次跑 6 个 baseline generation
2. 收集本次生成的 JSON 文件
3. 调用 scripts/step12_evaluate_pipeline.py --input <这些 json>
4. 更新 results/pipeline/pipeline_eval_summary.csv / json
5. 打印本次 model/dataset 的简表
```

历史规划命令如下；当前本地正式 pipeline 保留 Gemma2B 5090 最新结果，Qwen3-4B 已在  单独归档，Gemma2-9B 暂缓：

```bash
for MODEL in gemma2b qwen3_4b gemma9b; do
  DOLA_CUSTOM_GENERATE=$PWD/third_party/transformers-community-dola/custom_generate/generate.py \
  SKIP_EXISTING=1 \
  bash scripts/run_step11_greedy_baselines_with_eval.sh "$MODEL" all 0
done
```

这会得到每个方法在每个数据集上的 EM/F1：

```text
model × dataset × method → EM / F1
```

当前正式 summary 状态：

```text
Gemma2B: 42 rows
Qwen3-4B: waiting for 5090 results
Gemma2-9B: paused
```

---

#### 手动评测命令（备用）

如果只想评测已有输出，不重新生成：

```bash
# 评测所有 pipeline generation 输出
python scripts/step12_evaluate_pipeline.py --all

# 只评测某个模型
python scripts/step12_evaluate_pipeline.py --model gemma2b

# 只评测某个模型 + 某个数据集
python scripts/step12_evaluate_pipeline.py --model gemma2b --dataset nq
python scripts/step12_evaluate_pipeline.py --model gemma2b --dataset triviaqa
```

查看当前 summary：

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("results/pipeline/pipeline_eval_summary.csv")
cols = ["model", "dataset", "method", "n", "em", "f1"]
print(df[cols].sort_values(["model", "dataset", "method"]).to_string(index=False))
PY
```

---

#### 输出文件

每个 Step11 generation 文件对应一个 Step12 eval 文件：

```text
results/pipeline/{model}_{dataset}_{method}_alpha...json
results/pipeline/eval_{model}_{dataset}_{method}_alpha...json
```

summary 文件：

```text
results/pipeline/pipeline_eval_summary.csv
results/pipeline/pipeline_eval_summary.json
```

注意：`limit2` 文件只是 smoke test，不应进入正式主表。正式主表应使用不带 `_limit2` 的 full test set 结果。

---

#### 当前 Pipeline 结果摘要

完整表格位置：
```text
results/pipeline/pipeline_eval_summary.csv
results/pipeline/pipeline_eval_summary.json
results/all_finished_experiments.csv
```

Gemma2B 单独整理表：
```text
results/analysis/gemma2b_pipeline_results_table.csv
results/analysis/gemma2b_pipeline_results_table.md
```

Gemma2B 当前固定超参 plugin 结果（`alpha=0.5, tau=0.5`）：

| Dataset | Baseline | Plugin | EM baseline | EM plugin | ΔEM | F1 baseline | F1 plugin | ΔF1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NQ | vanilla_rag | csv_gated_vanilla_rag | 0.2576 | 0.2593 | +0.0017 | 0.3647 | 0.3625 | -0.0022 |
| NQ | cad_fixed | csv_gated_cad | 0.1701 | 0.2072 | +0.0371 | 0.2403 | 0.2915 | +0.0512 |
| NQ | acd | csv_gated_acd | 0.2499 | 0.2019 | -0.0479 | 0.3592 | 0.2977 | -0.0614 |
| NQ | context_ucd | csv_gated_context_ucd | 0.1133 | 0.1687 | +0.0554 | 0.1694 | 0.2441 | +0.0747 |
| NQ | dola | csv_gated_dola | 0.0562 | 0.1114 | +0.0551 | 0.1440 | 0.2073 | +0.0633 |
| TriviaQA | vanilla_rag | csv_gated_vanilla_rag | 0.4329 | 0.4926 | +0.0598 | 0.5183 | 0.5692 | +0.0509 |
| TriviaQA | cad_fixed | csv_gated_cad | 0.3058 | 0.4217 | +0.1159 | 0.3899 | 0.5038 | +0.1139 |
| TriviaQA | acd | csv_gated_acd | 0.4991 | 0.4947 | -0.0043 | 0.5760 | 0.5649 | -0.0111 |
| TriviaQA | context_ucd | csv_gated_context_ucd | 0.2029 | 0.3601 | +0.1573 | 0.2769 | 0.4405 | +0.1636 |
| TriviaQA | dola | csv_gated_dola | 0.1885 | 0.3425 | +0.1541 | 0.2890 | 0.4368 | +0.1478 |

结论：plugin 并非对所有方法单调提升，但对容易受坏检索伤害的 `cad_fixed`、`context_ucd` 和当前 HF DoLA 口径提升明显；TriviaQA 上整体提升更强。旧 `results/analysis/gemma2b_pipeline_results_table.*` 中的 DoLA baseline 属于过时分析表，不作为当前正式口径；当前正式结果以 `autodl_5090_2/results/pipeline/pipeline_eval_summary.csv` 为准。

当前最强 baseline：

| model | dataset | best baseline | EM | F1 |
|---|---|---|---:|---:|
| Gemma2-2B | NQ | vanilla_rag | 0.2576 | 0.3647 |
| Gemma2-2B | TriviaQA | acd | 0.4991 | 0.5760 |
| Qwen3-4B | NQ | acd | 0.2806 | 0.3712 |
| Qwen3-4B | TriviaQA | vanilla_rag | 0.3750 | 0.4667 |

说明：Gemma2-9B pipeline 暂未纳入当前最终 `autodl_5090_2/results/pipeline/pipeline_eval_summary.csv`，因此不列入这张最终 baseline 摘要。

解释：classifier gate 作为插件对“更依赖 retrieved context 且容易被坏检索拖垮”的方法更有效；当原方法本身已经较稳或 fallback 到 no-doc 会损失信息时，提升可能变小甚至为负。因此当前论文叙事聚焦：
```text
classifier as plugin:
  用 retrieved-doc classifier 判断是否启用某个已有解码/RAG方法，
  对可信 retrieved doc 使用原方法，对不可信 retrieved doc 回退到 no_doc 或更保守策略，
  从而减少坏检索伤害。
```

---

## 七、当前一句话总结

主线模型仍是 `Gemma2-2B + Gemma2-9B + Qwen3-4B-Base`。当前项目已经形成两条分类器线和一条端到端 pipeline 线：

```text
旧线：gold-vs-distractor CSV
  结论：证明 LLM hidden states / CSV 能极强地区分 gold evidence 与 distracting retrieved doc。
  状态：三模型 Step9b per-subset 已完成，可作为论文 separability / amplification 证据。

新线：retrieved-doc CSV
  结论目标：判断真实 RAG pipeline 中 top-1 retrieved_doc 是否值得信任。
  状态：作为 Step11 gate 主线，Gemma2B CSV_R best=`inject_14_cls_18` 已用于 pipeline；Qwen3-4B CSV_R best=`inject_21_cls_22` 已整理到 pipeline 专用目录。

Pipeline 线：greedy baseline + classifier-gated plugin 方法
  结论目标：评估不同 RAG / contrastive decoding 方法在 NQ 和 TriviaQA 上的 EM/F1。
  状态：Gemma2B 与 Qwen3-4B 均已完成 6 个 baseline 与 5 个 plugin 的 tau=0.5/0.4/0.3 对照。
```

**当前进度**：

- 三模型 Step 6/6b/7/8 全部完成（Step 8 四种 classifier：LR / MLP / Centroid / Mass-Mean）。
- Step 8b（旧 gold-vs-distractor DPR encoder-dot baseline）完成：combined AUROC=0.2919，反相关，是关键论文素材。
- Step 8R（新 retrieved-doc DPR retriever-score baseline）完成：combined AUROC=0.6376，Acc=0.6253，结果 `results/probe_retrieved/dpr_baseline.json`；该口径使用 `doc.score`，不要与旧 DPR encoder-dot baseline 混用。
- Step 8 gold-vs-distractor 外部 baseline 新增 CRAG retrieval evaluator：
  - 代码：`scripts/step8_crag_baseline.py --mode gold_distractor`
  - 模型路径：`CRAG/model`
  - 结果：`results/probe/crag_gold_distractor_baseline_results.json`
  - combined AUROC=0.4519，Acc=0.5000；NQ AUROC=0.4393；TriviaQA AUROC=0.4631。
- Step 8R 外部 baseline 新增 CRAG retrieval evaluator：
  - 代码：`scripts/step8_crag_baseline.py`
  - 模型路径：`CRAG/model`
  - 结果：`results/probe_retrieved/crag_baseline_results.json`
  - combined AUROC=0.5171，Acc=0.4244，明显弱于 retrieved-doc hidden-state probe / CSV。
- Step 8R 外部 baseline 新增 Self-RAG retrieval evaluator：
  - 代码：`scripts/step8_selfrag_baseline.py`
  - 模型路径：`/path/to/selfrag_llama2_13b`
  - 结果：`results/probe_retrieved/selfrag_baseline_results.json`
  - combined AUROC=0.7688，Acc=0.6984；强于 CRAG，但仍弱于 Gemma2B Step9R CSV。
- Step 8 gold-vs-distractor 外部 baseline 新增 Self-RAG retrieval evaluator：
  - 代码：`scripts/step8_selfrag_baseline.py --mode gold_distractor`
  - 结果：`results/probe/selfrag_gold_distractor_baseline_results.json`
  - combined AUROC=0.6451，Acc=0.6046。
- 旧 Step9 gold-vs-distractor：
  - Gemma2B sweep + Step9b 完成，best AUROC≈0.9964，inject=1, cls=16。
  - Gemma9B current sweep + Step9b 完成，38 configs，best AUROC≈0.9974，inject=0, cls=30。
  - Qwen3-4B current sweep + Step9b 完成，24 configs，best AUROC≈0.9961，inject=6, cls=24。
  - 这条线作为 separability / CSV amplification 证据保留，不作为 Step11 gate 主线。
- 新 Step9R retrieved-doc：
  - `data/final_retrieved/` 构建完成，train 平衡，eval 自然分布。
  - `step9_train_csv_retrieved.py` 已固定 seed=42。
  - Gemma2B fixed-layer 调参完成：`lr=0.002, ema=0.99` 最好。
  - Gemma2B expanded sweep 已完成：`inject=1,2,3,4,5,6,10,12,16`，`cls=8,12,16,18,20,24,-1`，共 57 个有效 config。
  - Gemma2B 又补跑固定 `CLS=18` inject sweep：`cls=18`，`inject=0..17`，该层切片共 18 个 config；若接在 57-config sweep 后运行，最终总 sweep JSON 预计新增约 9 个 unique config。
  - Gemma2B retrieved-doc gate best 已确认：`inject_14_cls_18`，AUROC=0.902907，Acc=0.837471，best_epoch=7。注意这不同于旧 gold-vs-distractor Step9 的 0.9964；Step9R 是真实 retrieved top-1 分布，难度更高。
  - Qwen3-4B fixed-layer 调参完成：固定 `inject=4, cls=30` 时选择 `lr=0.002, ema=0.99`。
  - Qwen3-4B expanded sweep 已完成：`inject=2,4,6,8,10,11`，`cls=8,12,16,20,22,23,24,27,30,-1`，共 57 个有效 config。
  - Qwen3-4B 固定 `CLS=22` inject sweep 与 hparam grid 已完成；当前用于 pipeline 的 best=`inject_21_cls_22`，AUROC=0.915867，Acc=0.855269，best_epoch=20。
  - Qwen3-4B pipeline best 目录：`results/csv_retrieved_qwen3_4b_best_for_pipeline/`。
- Step 10 Logit Lens 完成：三模型完整 eval 均显示 `gold_ctx > no_ctx ≈ dis_ctx`，peak layer 稳定出现在后层约 86%-88% 深度。
- Step 10R retrieved-doc Logit Lens 新增：
  - Gemma2B/NQ 完成，真实 retrieved relevant 的 peak layer 仍为 23。
  - 输出：`results/logit_lens_retrieved/gemma2b_retrieved_logit_lens_nq.json`。
- Step 10b 画图完成：三模型 `logp_by_layer`、`delta_by_layer`、`logp_by_subset` 九张图已生成。
- Step 11 已更新到 greedy-only pipeline：
  - baseline 不使用 CSV；Gemma2B 与 Qwen3-4B gated 方法均已用各自 Step9R best checkpoint 完成 pipeline。
  - baseline 方法已扩展为 `no_doc / vanilla_rag / cad_fixed / acd / context_ucd / dola`。
  - `context_ucd` 是同模型 doc/no-doc 的 Context-UCD-Energy / UCD-style energy-weighted contrast，不是原始 UCD。
  - classifier plugin 主线对每个使用 retrieved document 的 baseline 加 gate：`csv_gated_vanilla_rag / csv_gated_cad / csv_gated_acd / csv_gated_context_ucd / csv_gated_dola`；Gemma2B 与 Qwen3-4B 均已跑 tau=0.5/0.4/0.3。
  - CAD 相关主实验固定 `alpha=0.5`，与 CAD 论文默认口径对齐，减少额外调参自由度。
  - Gemma2B `tau=0.5/0.4/0.3` plugin 结果已完成并同步到本地 `results/pipeline/`；Qwen3-4B 对应结果已同步到 `autodl_5090_2/results/pipeline/`。
  - DoLA 使用本地 `third_party/transformers-community-dola/custom_generate/generate.py`，通过 `DOLA_CUSTOM_GENERATE` 指定。
  - 三模型默认使用本地 snapshot 路径，避免运行时下载模型。
- Step 12 已实现并修复 answers 字符串 list 解析问题；summary 现在从所有 durable `eval_*.json` 重建，避免单次评测覆盖成局部结果。

**Qwen3-4B pipeline 历史运行命令（5090 服务器，已完成）**：

```bash
MAX_INPUT_LENGTH=512 CAD_ALPHA=0.5 SKIP_EXISTING=1 \
bash scripts/run_step11_greedy_baselines_with_eval.sh qwen3_4b all 0

for METHOD in csv_gated_vanilla_rag csv_gated_cad csv_gated_acd csv_gated_context_ucd csv_gated_dola; do
  METHOD="$METHOD" \
  ALPHAS="0.5" \
  TAUS="0.5 0.4 0.3" \
  MAX_INPUT_LENGTH=512 \
  SKIP_EXISTING=1 \
  bash scripts/run_step11_cad_alpha_sweep_with_eval.sh qwen3_4b all 0
done
```

对应实验：

```text
model=qwen3_4b
task=Step11/12 pipeline
CSV_R best=inject_21_cls_22
expected generation files=42
expected eval files=42
local status=已回传到 autodl_5090_2/results/pipeline/；summary 总表=84 eval rows，其中 qwen3_4b=42 rows
```

**已确立的核心论文叙事**：

```text
DPR gold-vs-distractor dot:  AUROC≈0.29，反相关
DPR retrieved-doc score:        AUROC≈0.64，弱于 Self-RAG 和 LLM hidden-state probe / CSV
Self-RAG evaluator:             retrieved-doc AUROC≈0.7688，强于 CRAG 但弱于 LLM hidden-state probe / CSV
LR/MLP/Mass-Mean probe:         AUROC≈0.74-0.90，说明 LLM hidden states 有文档质量信号
旧 CSV gold-vs-distractor:      三模型 AUROC≈0.996-0.997，证明轻量 steering 可强力放大该信号
Logit Lens:                     gold_ctx > no_ctx≈dis，说明 useful 文档确实增强答案概率信号
Pipeline baseline:              no_doc / vanilla_rag / CAD / ACD / Context-UCD-Energy / DoLA 的 EM/F1 已完成
Pipeline plugin:                把 retrieved-doc classifier 接到 vanilla_rag / CAD / ACD / Context-UCD / DoLA 前；可信则运行原 baseline，不可信则退回 no_doc
新 Step9R retrieved-doc CSV:    Gemma2B gate checkpoint 已用于 pipeline；Qwen3-4B best=`inject_21_cls_22` 已用于 pipeline，结果已回传
```

**下一步**：

1. Qwen3-4B pipeline 已回传并整理；后续如需释放磁盘，可在确认无误后删除旧备份 `autodl_5090_1`。
2. 决定是否补 Gemma2-9B CSV_R sweep；若论文需要三模型 retrieved-doc CSV_R 完整性，则优先补。
3. 等 2B / 4B pipeline 结论稳定后，再决定是否跑 Gemma2-9B pipeline。
4. 若要写论文主表，优先从 `results/analysis/pipeline_plugin_comparison_by_tau.md` 选 best-tau 或固定 tau=0.5 两种口径。
