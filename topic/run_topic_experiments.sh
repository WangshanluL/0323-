#!/usr/bin/env bash
set -euo pipefail

# 默认：单次主流程（1×embed → 1×cluster → 1×score），与实地落地一致。
# 离线超参网格请使用：bash ./run_topic_grid.sh
#
# 调试时可追加小样本参数（不跑全量账号/发言），例如：
#   --max_align_accounts 64
#   --max_text_rows_source 800 --max_text_rows_target 800
#
# 请先在本终端手动激活目标 Python 环境（例如：conda activate topic），再运行本脚本。
# 可选：覆盖解释器路径，例如 PYTHON_BIN=/path/to/python bash ./run_topic_experiments.sh

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[Info] Python: ${PYTHON_BIN}"
echo "[Run] Single-pass (bidirectional window, soft assignment)..."
"${PYTHON_BIN}" "./topic_theme_match.py" \
  --device cuda \
  --topk 10 \
  --time_window_days 2 \
  --window_direction bidirectional \
  --assignment_mode soft \
  --softmax_temperature 0.07 \
  --topic_fit_scope source_only \
  --num_topics 80 \
  --cluster_backend kmeans \
  --topic_weight 0.7 \
  --embed_weight 0.3 \
  --global_weight 0.1 \
  --day_match_agg topk_mean \
  --day_match_topk 2 \
  --candidate_delim ";" \
  --output_csv "topic_pred_top10_bidirectional.csv"

echo "[Done] Output: topic_pred_top10_bidirectional.csv"
echo "[Hint] 海量消息可改用 --cluster_backend minibatch 缩短聚类耗时（需在小样本上先验指标）。"
