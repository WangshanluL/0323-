"""
基于用户画像（User Picture / Profile）的用户对齐方法
====================================================
读取预生成的 source / target 用户画像 CSV，通过画像标签相似度
对每个 source 用户生成 top-K 候选 target 用户列表。

画像 CSV 由 picture.py 生成，包含 profile_json 列（JSON 字符串）。
相似度计算：标签集合的 Jaccard 相似 + emotion_tone / content_language 精确匹配加分。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


# ============================================================
# 画像解析
# ============================================================

def _extract_label_set(profile: Dict[str, Any]) -> Set[str]:
    """从画像 JSON 中提取所有标签（topic + lifestyle + language_style）为扁平集合。"""
    labels: Set[str] = set()
    for key in ("topic_labels", "lifestyle_labels", "language_style_labels"):
        val = profile.get(key, {})
        if isinstance(val, dict):
            for cat_labels in val.values():
                if isinstance(cat_labels, list):
                    for lbl in cat_labels:
                        lbl_s = str(lbl).strip()
                        if lbl_s:
                            labels.add(lbl_s)
        elif isinstance(val, list):
            for lbl in val:
                lbl_s = str(lbl).strip()
                if lbl_s:
                    labels.add(lbl_s)
    return labels


def _parse_profile(raw: str) -> Optional[Dict[str, Any]]:
    """安全解析 profile_json 字段。"""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _load_profiles(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    读取画像 CSV，返回 {account_id: profile_dict}。
    CSV 需要 account_id 和 profile_json 两列。
    """
    df = pd.read_csv(csv_path)
    col_map = {c.strip().lower(): c for c in df.columns}

    # 找 account_id 列
    aid_col = None
    for candidate in ("account_id", "accountid", "user_id", "userid"):
        if candidate in col_map:
            aid_col = col_map[candidate]
            break
    if aid_col is None:
        aid_col = df.columns[0]

    # 找 profile_json 列
    pj_col = None
    for candidate in ("profile_json", "profile"):
        if candidate in col_map:
            pj_col = col_map[candidate]
            break
    if pj_col is None:
        raise ValueError(f"画像 CSV 缺少 profile_json 列: {csv_path}，列: {df.columns.tolist()}")

    profiles: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        aid = str(row[aid_col]).strip()
        parsed = _parse_profile(str(row[pj_col]))
        if parsed is not None:
            profiles[aid] = parsed
    return profiles


# ============================================================
# 相似度计算
# ============================================================

def _similarity(src_profile: Dict[str, Any], tgt_profile: Dict[str, Any]) -> float:
    """
    计算两个用户画像之间的相似度分数（0~1 范围）。
    - 标签 Jaccard 相似度（权重 0.7）
    - emotion_tone 精确匹配（权重 0.15）
    - content_language 精确匹配（权重 0.15）
    """
    src_labels = _extract_label_set(src_profile)
    tgt_labels = _extract_label_set(tgt_profile)

    # Jaccard
    if src_labels or tgt_labels:
        intersection = src_labels & tgt_labels
        union = src_labels | tgt_labels
        jaccard = len(intersection) / len(union) if union else 0.0
    else:
        jaccard = 0.0

    # emotion_tone
    src_emo = str(src_profile.get("emotion_tone", "")).strip()
    tgt_emo = str(tgt_profile.get("emotion_tone", "")).strip()
    emo_match = 1.0 if (src_emo and src_emo == tgt_emo) else 0.0

    # content_language
    src_lang = str(src_profile.get("content_language", "")).strip()
    tgt_lang = str(tgt_profile.get("content_language", "")).strip()
    lang_match = 1.0 if (src_lang and src_lang == tgt_lang) else 0.0

    return 0.7 * jaccard + 0.15 * emo_match + 0.15 * lang_match


# ============================================================
# 对齐主函数
# ============================================================

def run_alignment(
    source_file: str,
    target_file: str,
    need_align_file: str,
    output_file: str,
    K: int,
    ground_truth_file: str,
    *,
    source_profile_file: str = "",
    target_profile_file: str = "",
) -> Tuple[str, Dict]:
    """
    基于用户画像的对齐方法。

    Parameters
    ----------
    source_file        : 源平台消息 CSV（本方法不直接使用，保持接口统一）
    target_file        : 目标平台消息 CSV（本方法不直接使用）
    need_align_file    : 待对齐 source 账号列表 CSV（含 account_id 列）
    output_file        : 输出结果 CSV 路径
    K                  : Top-K 候选数量
    ground_truth_file  : 真值对应表 CSV
    source_profile_file: 源平台用户画像 CSV（由 picture.py 生成）
    target_profile_file: 目标平台用户画像 CSV（由 picture.py 生成）

    Returns
    -------
    (output_file_path, metrics_dict)
    """
    if not source_profile_file or not target_profile_file:
        raise ValueError("UserPicture 方法需要指定 --source_profile_file 和 --target_profile_file")

    # 1. 加载画像
    print("[UserPicture] 加载源平台画像...")
    src_profiles = _load_profiles(source_profile_file)
    print(f"[UserPicture] 源平台画像数量: {len(src_profiles)}")

    print("[UserPicture] 加载目标平台画像...")
    tgt_profiles = _load_profiles(target_profile_file)
    print(f"[UserPicture] 目标平台画像数量: {len(tgt_profiles)}")

    # 2. 读取待对齐账号列表
    need_df = pd.read_csv(need_align_file)
    col_lower = {c.strip().lower(): c for c in need_df.columns}
    aid_col = next(
        (col_lower[k] for k in ("account_id", "accountid", "user_id") if k in col_lower),
        need_df.columns[0],
    )
    source_ids = [str(x).strip() for x in need_df[aid_col].dropna().unique()]
    print(f"[UserPicture] 待对齐源用户数: {len(source_ids)}")

    # 3. 对每个 source 用户计算与所有 target 用户的相似度，取 top-K
    tgt_ids = list(tgt_profiles.keys())
    results = []

    for src_id in source_ids:
        src_prof = src_profiles.get(src_id)
        if src_prof is None:
            results.append({"source_account_id": src_id, "predict_target_account_id": ""})
            continue

        scores = []
        for tgt_id in tgt_ids:
            tgt_prof = tgt_profiles[tgt_id]
            sim = _similarity(src_prof, tgt_prof)
            scores.append((tgt_id, sim))

        scores.sort(key=lambda x: -x[1])
        top_k = [tid for tid, _ in scores[:K]]
        results.append({
            "source_account_id": src_id,
            "predict_target_account_id": ",".join(top_k),
        })

    # 4. 写出结果
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[UserPicture] 结果已写入: {output_file}")

    # 5. 计算指标
    gt_df = pd.read_csv(ground_truth_file)
    gt_cols_lower = {c.strip().lower(): c for c in gt_df.columns}
    gt_src_col = next(
        (gt_cols_lower[k] for k in ("source_account_id", "source_account", "source") if k in gt_cols_lower),
        gt_df.columns[0],
    )
    gt_tgt_col = next(
        (gt_cols_lower[k] for k in ("target_account_id", "target_account", "target") if k in gt_cols_lower),
        gt_df.columns[1],
    )
    gt_map = {str(r[gt_src_col]): str(r[gt_tgt_col]) for _, r in gt_df.iterrows()}

    hit1, hitK, mrr_sum, cnt = 0.0, 0.0, 0.0, 0
    for row in results:
        src = row["source_account_id"]
        if src not in gt_map:
            continue
        true_tgt = gt_map[src]
        preds = [p.strip() for p in row["predict_target_account_id"].split(",") if p.strip()]
        cnt += 1
        rank = next((i + 1 for i, p in enumerate(preds) if p == true_tgt), None)
        if rank is not None:
            if rank == 1:
                hit1 += 1
            if rank <= K:
                hitK += 1
            mrr_sum += 1.0 / rank

    metrics = {
        "hit@1": hit1 / cnt if cnt else 0.0,
        f"hit@{K}": hitK / cnt if cnt else 0.0,
        "MRR": mrr_sum / cnt if cnt else 0.0,
        "evaluated_users": cnt,
    }
    print(f"[UserPicture] 评估: hit@1={metrics['hit@1']:.4f}, hit@{K}={metrics[f'hit@{K}']:.4f}, MRR={metrics['MRR']:.4f}")

    return output_file, metrics
