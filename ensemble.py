"""
投票融合（Borda Count）：将多路对齐结果（GNN / Style / Topic）合并为最终 top-K 候选。

典型用法：
  python ensemble.py \\
    --result_csvs gnn_out.csv style_out.csv topic_out.csv \\
    --output_file ensemble_out.csv \\
    --k 10 \\
    --ground_truth_file data/ground_truth_mapping.csv

或在 Python 中直接调用：
  from ensemble import run_voting_ensemble
  output_path, metrics = run_voting_ensemble(
      result_csvs=["gnn_out.csv", "style_out.csv", "topic_out.csv"],
      output_file="ensemble_out.csv",
      K=10,
      ground_truth_file="data/ground_truth_mapping.csv",
  )
"""
from __future__ import annotations

import argparse
import ast
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _read_alignment_csv(csv_path: str) -> Dict[str, List[str]]:
    """
    读取两列对齐结果 CSV，返回 {source_id: [candidate1, candidate2, ...]}。
    兼容多种格式：
    - GNN 格式：source_account_id, predict_target_account_id（逗号分隔字符串）
    - style 格式：source_user, aligned_candidates（逗号/分号分隔或 Python 列表字面量）
    - topic 格式：source_account.id, predict.target.account.id（分号分隔字符串）
    """
    df = pd.read_csv(csv_path)

    src_col: Optional[str] = None
    tgt_col: Optional[str] = None
    for c in df.columns:
        cl = c.strip().lower().replace(".", "_").replace(" ", "_")
        if cl in ("source_account_id", "source_user", "source_account", "source"):
            src_col = c
        elif cl in ("predict_target_account_id", "aligned_candidates", "target_account_id", "target"):
            tgt_col = c

    if src_col is None or tgt_col is None:
        if len(df.columns) >= 2:
            src_col, tgt_col = df.columns[0], df.columns[1]
        else:
            raise ValueError(f"无法识别 CSV 列名：{csv_path}，列：{df.columns.tolist()}")

    result: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        src = str(row[src_col])
        raw = row[tgt_col]
        if isinstance(raw, list):
            preds = [str(p) for p in raw]
        elif isinstance(raw, str):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    preds = [str(p) for p in parsed]
                else:
                    preds = [str(parsed)]
            except (ValueError, SyntaxError):
                delim = ";" if ";" in raw else ","
                preds = [p.strip() for p in raw.split(delim) if p.strip()]
        else:
            preds = []
        result[src] = preds

    return result


def _borda_count_merge(
    candidate_lists: List[Dict[str, List[str]]],
    K: int,
) -> Dict[str, List[str]]:
    """
    对多路候选列表执行 Borda count 融合。
    排名第 i（1-indexed）得分 = K - i + 1（i <= K），超出 K 的候选得 0 分。
    汇总各方法得分后，取得分最高的 top-K 作为最终候选。
    """
    all_sources: set = set()
    for clist in candidate_lists:
        all_sources.update(clist.keys())

    merged: Dict[str, List[str]] = {}
    for src in sorted(all_sources):
        scores: Dict[str, float] = {}
        for clist in candidate_lists:
            for rank_0, tgt in enumerate(clist.get(src, [])):
                score = max(K - rank_0, 0)
                if score > 0:
                    scores[tgt] = scores.get(tgt, 0.0) + score
        ranked = sorted(scores.keys(), key=lambda t: -scores[t])[:K]
        merged[src] = ranked

    return merged


def _compute_metrics(
    merged: Dict[str, List[str]],
    ground_truth_file: str,
    K: int,
) -> Dict:
    """对比真值表计算 hit@1, hit@K, MRR。"""
    gt_df = pd.read_csv(ground_truth_file)
    cols_lower = {c.strip().lower(): c for c in gt_df.columns}
    src_col = next(
        (cols_lower[k] for k in ("source_account_id", "source_account", "source") if k in cols_lower),
        gt_df.columns[0],
    )
    tgt_col = next(
        (cols_lower[k] for k in ("target_account_id", "target_account", "target") if k in cols_lower),
        gt_df.columns[1],
    )
    gt_map = {str(r[src_col]): str(r[tgt_col]) for _, r in gt_df.iterrows()}

    hit1_sum = 0.0
    hitK_sum = 0.0
    mrr_sum = 0.0
    cnt = 0

    for src, preds in merged.items():
        if src not in gt_map:
            continue
        true_tgt = gt_map[src]
        cnt += 1
        rank = None
        for i, p in enumerate(preds):
            if str(p) == true_tgt:
                rank = i + 1
                break
        if rank is not None:
            if rank == 1:
                hit1_sum += 1.0
            if rank <= K:
                hitK_sum += 1.0
            mrr_sum += 1.0 / rank

    if cnt == 0:
        return {"hit@1": 0.0, f"hit@{K}": 0.0, "MRR": 0.0, "evaluated_users": 0}
    return {
        "hit@1": hit1_sum / cnt,
        f"hit@{K}": hitK_sum / cnt,
        "MRR": mrr_sum / cnt,
        "evaluated_users": cnt,
    }


def run_voting_ensemble(
    result_csvs: List[str],
    output_file: str,
    K: int = 10,
    ground_truth_file: Optional[str] = None,
) -> Tuple[str, Dict]:
    """
    Borda count 投票融合多路对齐结果。

    Parameters
    ----------
    result_csvs       : 各方法输出的 CSV 文件路径列表（至少 2 个）
    output_file       : 融合结果输出 CSV 路径
    K                 : Top-K 候选数量
    ground_truth_file : 真值表 CSV 路径（可选）

    Returns
    -------
    (output_file_path, metrics)
      metrics 含 hit@1, hit@{K}, MRR, evaluated_users；若无真值则为 {}
    """
    if not result_csvs:
        raise ValueError("result_csvs 不能为空")

    print(f"[ensemble] 融合 {len(result_csvs)} 路结果（Borda Count, K={K}）...")
    candidate_lists: List[Dict[str, List[str]]] = []
    for path in result_csvs:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"结果 CSV 不存在: {path}")
        cands = _read_alignment_csv(path)
        candidate_lists.append(cands)
        print(f"  读入: {path}  ({len(cands)} 个 source 账号)")

    merged = _borda_count_merge(candidate_lists, K=K)
    print(f"[ensemble] 融合完成，共 {len(merged)} 个 source 账号。")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    rows = [
        {"source_account_id": src, "predict_target_account_id": ",".join(preds)}
        for src, preds in merged.items()
    ]
    pd.DataFrame(rows).to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[ensemble] 已写入: {output_file}")

    metrics: Dict = {}
    if ground_truth_file and os.path.isfile(ground_truth_file):
        metrics = _compute_metrics(merged, ground_truth_file, K)
        print(
            f"[ensemble] 评估结果: "
            f"hit@1={metrics['hit@1']:.4f}, "
            f"hit@{K}={metrics[f'hit@{K}']:.4f}, "
            f"MRR={metrics['MRR']:.4f}, "
            f"evaluated_users={metrics['evaluated_users']}"
        )

    return output_file, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Borda count 投票融合多路对齐结果。")
    parser.add_argument(
        "--result_csvs",
        type=str,
        nargs="+",
        required=True,
        help="各方法输出的 CSV 路径（至少 2 个），空格分隔。",
    )
    parser.add_argument("--output_file", type=str, required=True, help="融合结果输出 CSV 路径。")
    parser.add_argument("--k", type=int, default=10, help="Top-K 候选数量。")
    parser.add_argument("--ground_truth_file", type=str, default=None, help="真值表 CSV 路径（可选）。")

    args = parser.parse_args()
    run_voting_ensemble(
        result_csvs=args.result_csvs,
        output_file=args.output_file,
        K=args.k,
        ground_truth_file=args.ground_truth_file,
    )
