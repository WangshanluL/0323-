"""
对齐流水线共用工具：读消息、相似度、Top-K、评估、两列 CSV 写出、真值表加载。
"""
from __future__ import annotations

import ast
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def load_messages_from_path(path: str) -> pd.DataFrame:
    """
    支持单个 CSV 或目录下多个 *.csv（无 account_id 列时用文件名作为 account_id）。
    """
    if os.path.isdir(path):
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        dfs = []
        for fp in csv_files:
            df = pd.read_csv(fp)
            if "account_id" not in df.columns:
                acc_id = os.path.splitext(os.path.basename(fp))[0]
                df["account_id"] = acc_id
            dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, axis=0, ignore_index=True)

    if os.path.isfile(path):
        return pd.read_csv(path)

    raise FileNotFoundError(f"Input path not found: {path}")


def build_similarity_matrix(
    source_features: Dict[str, np.ndarray],
    target_features: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str], List[str]]:
    source_ids = list(source_features.keys())
    target_ids = list(target_features.keys())
    X = np.stack([source_features[u] for u in source_ids], axis=0)
    Y = np.stack([target_features[u] for u in target_ids], axis=0)
    X = normalize(X)
    Y = normalize(Y)
    sim = cosine_similarity(X, Y)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    return sim, source_ids, target_ids


def get_topk_candidates(
    sim_matrix: np.ndarray,
    source_ids: List[str],
    target_ids: List[str],
    k: int,
) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, sim_matrix.shape[1])
    rows = []
    for i, src_id in enumerate(source_ids):
        sims = sim_matrix[i]
        topk_idx = np.argsort(-sims)[:k]
        topk_ids = [target_ids[j] for j in topk_idx]
        rows.append(
            {
                "source_account_id": src_id,
                "predict_target_account_id": topk_ids,
            }
        )
    return pd.DataFrame(rows)


def compute_topk_from_feature_dicts(
    source_features: Dict[str, np.ndarray],
    target_features: Dict[str, np.ndarray],
    k: int,
) -> pd.DataFrame:
    sim, src_ids, tgt_ids = build_similarity_matrix(source_features, target_features)
    return get_topk_candidates(sim, src_ids, tgt_ids, k=k)


def load_ground_truth_align_accounts(gt_path: str) -> pd.DataFrame:
    """
    读取 align_accounts.csv（或同结构真值表），统一为列 source_account_id, target_account_id。
    支持：
    - 已有 source_account_id / target_account_id
    - 仅两列时按顺序视为 source / target
    - 常见别名：wx/dy、wechat/douyin 等
    """
    df = pd.read_csv(gt_path)
    if df.empty:
        return df

    col_lower = {c: str(c).strip().lower() for c in df.columns}
    inv = {v: k for k, v in col_lower.items()}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in inv:
                return inv[n]
        return None

    s_col = pick("source_account_id", "source", "wx_account", "wechat", "wx_id", "weixin")
    t_col = pick("target_account_id", "target", "dy_account", "douyin", "dy_id", "tiktok")

    if s_col and t_col:
        out = pd.DataFrame(
            {
                "source_account_id": df[s_col].astype(str),
                "target_account_id": df[t_col].astype(str),
            }
        )
        return out

    if len(df.columns) == 2:
        c0, c1 = df.columns[0], df.columns[1]
        return pd.DataFrame(
            {
                "source_account_id": df[c0].astype(str),
                "target_account_id": df[c1].astype(str),
            }
        )

    raise ValueError(
        f"Cannot infer source/target columns from {gt_path}. "
        "Expected source_account_id & target_account_id, or two columns."
    )


def write_alignment_csv_two_columns(
    candidates_df: pd.DataFrame,
    out_path: str,
) -> None:
    """
    两列 CSV：第一列 source 侧全部用户；第二列为 top-k 候选列表（与 predict_target_account_id 一致）。
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out = pd.DataFrame(
        {
            "source_user": candidates_df["source_account_id"].astype(str),
            "aligned_candidates": candidates_df["predict_target_account_id"],
        }
    )
    out.to_csv(out_path, index=False, encoding="utf-8-sig")


def compute_alignment_metrics(
    candidates_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    k: int,
) -> Dict[str, float]:
    """
    与 align_accounts 真值对比，输出 hit@1、hit@k、MRR。
    candidates 需含 source_account_id, predict_target_account_id（可为列表或字符串形式的列表）。
    """
    gt_map = {
        str(row["source_account_id"]): str(row["target_account_id"])
        for _, row in gt_df.iterrows()
    }
    hit1 = 0.0
    hitk = 0.0
    mrr_sum = 0.0
    cnt = 0
    for _, row in candidates_df.iterrows():
        src = str(row["source_account_id"])
        if src not in gt_map:
            continue
        true_tgt = gt_map[src]
        preds = row["predict_target_account_id"]
        if isinstance(preds, str):
            try:
                preds = ast.literal_eval(preds)
            except Exception:
                preds = []
        cnt += 1
        rank = None
        for idx, p in enumerate(preds):
            if str(p) == true_tgt:
                rank = idx + 1
                break
        if rank is not None:
            if rank == 1:
                hit1 += 1.0
            if rank <= k:
                hitk += 1.0
            mrr_sum += 1.0 / rank
    if cnt == 0:
        return {"hit@1": 0.0, f"hit@{k}": 0.0, "MRR": 0.0, "evaluated_users": 0}
    return {
        "hit@1": hit1 / cnt,
        f"hit@{k}": hitk / cnt,
        "MRR": mrr_sum / cnt,
        "evaluated_users": cnt,
    }


def metrics_from_two_column_csv(candidates_csv: str, gt_df: pd.DataFrame, k: int) -> Dict[str, float]:
    """从两列输出 CSV 读回并算指标（列名 source_user / aligned_candidates）。"""
    df = pd.read_csv(candidates_csv)
    if "source_account_id" not in df.columns:
        df = df.rename(
            columns={"source_user": "source_account_id", "aligned_candidates": "predict_target_account_id"}
        )
    return compute_alignment_metrics(df, gt_df, k=k)
