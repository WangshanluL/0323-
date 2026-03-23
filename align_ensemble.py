#!/usr/bin/env python3
"""
三路并行用户对齐 + Borda Count 投票融合
========================================
并行运行 GNN / UserPicture / Topic 三种对齐方法，对每个 source 用户的候选列表做
Borda Count 投票，输出最终 top-K 融合结果 CSV 并打印评估指标。

输入 (4 个 CSV 文件 + 2 个画像 CSV):
  --source_file           : 源平台消息 CSV（含 account_id, room_id, msg, time, type 列）
  --target_file           : 目标平台消息 CSV
  --need_align_file       : 待对齐 source 账号列表 CSV（含 account_id 列）
  --ground_truth_file     : 真值对应表 CSV（source_account, target_account 两列）
  --source_profile_file   : 源平台用户画像 CSV（由 picture.py 生成）
  --target_profile_file   : 目标平台用户画像 CSV（由 picture.py 生成）

输出:
  --output_file       : 投票融合后的 top-K 结果 CSV（两列：source_account_id, predict_target_account_id）
  控制台打印每个方法及最终融合的 hit@1 / hit@K / MRR

典型用法:
  python align_ensemble.py \\
      --source_file   data/source_alignment_ground_truth.csv \\
      --target_file   data/target_alignment_ground_truth.csv \\
      --need_align_file data/source_need_align_accounts.csv \\
      --ground_truth_file data/ground_truth_mapping.csv \\
      --source_profile_file data/source_profile.csv \\
      --target_profile_file data/target_profile.csv \\
      --output_file   results/ensemble_out.csv \\
      --k 10

仅运行 GNN + UserPicture（跳过 Topic 的重型 embedding）:
  python align_ensemble.py ... --disable_topic
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 内部：动态加载各子模块（importlib 方式，避免 sys.path 全局污染）
# ============================================================
_PROJ_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name: str, abs_path: str):
    """
    用 importlib 动态加载模块。
    将模块注册到 sys.modules[name]，使模块内部的兄弟模块 import 可以生效。
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, abs_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # 先注册，防止循环 import
    spec.loader.exec_module(mod)     # 执行模块代码（会触发内部 import）
    return mod


def _get_runner_gnn():
    """加载 GNN/alignv3.py 并返回 run_alignment 函数。"""
    mod = _load_module("alignv3", os.path.join(_PROJ_DIR, "GNN", "alignv3.py"))
    return mod.run_alignment


def _get_runner_user_picture():
    """加载 user_picture/user_picture_align.py 并返回 run_alignment 函数。"""
    mod = _load_module(
        "user_picture_align",
        os.path.join(_PROJ_DIR, "user_picture", "user_picture_align.py"),
    )
    return mod.run_alignment


def _get_runner_topic():
    """加载 topic/topic_theme_match.py 并返回 run_alignment 函数。"""
    mod = _load_module(
        "topic_theme_match",
        os.path.join(_PROJ_DIR, "topic", "topic_theme_match.py"),
    )
    return mod.run_alignment


# ============================================================
# 候选列表解析
# ============================================================
def _parse_candidates(raw) -> List[str]:
    """将 CSV 字段（字符串 / Python 列表字面量 / 分号或逗号分隔）解析为字符串列表。"""
    if isinstance(raw, list):
        return [str(p) for p in raw]
    if not isinstance(raw, str):
        return []
    raw = raw.strip()
    # 尝试 Python 列表字面量：['a', 'b', ...]
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(p) for p in parsed]
    except (ValueError, SyntaxError):
        pass
    # 分号或逗号分隔
    delim = ";" if ";" in raw else ","
    return [p.strip() for p in raw.split(delim) if p.strip()]


def _read_result_csv(csv_path: str) -> Dict[str, List[str]]:
    """
    读取对齐结果 CSV，返回 {source_id: [cand1, cand2, ...]}。
    兼容 GNN / Style / Topic 三种列名格式。
    """
    df = pd.read_csv(csv_path)
    src_col = tgt_col = None
    for c in df.columns:
        cl = c.strip().lower().replace(".", "_").replace(" ", "_")
        if cl in ("source_account_id", "source_user", "source_account", "source"):
            src_col = c
        elif cl in (
            "predict_target_account_id",
            "aligned_candidates",
            "target_account_id",
            "target",
        ):
            tgt_col = c
    if src_col is None or tgt_col is None:
        if len(df.columns) >= 2:
            src_col, tgt_col = df.columns[0], df.columns[1]
        else:
            raise ValueError(f"无法识别 CSV 列名：{csv_path}，列：{df.columns.tolist()}")
    return {
        str(row[src_col]): _parse_candidates(row[tgt_col])
        for _, row in df.iterrows()
    }


# ============================================================
# Borda Count 融合
# ============================================================
def borda_count_merge(
    candidate_dicts: List[Dict[str, List[str]]],
    K: int,
) -> Dict[str, List[str]]:
    """
    Borda Count 排名融合。

    对每个 source 用户，遍历所有方法的候选列表：
    - 排名第 i（1-indexed）得分 = K - i + 1（i <= K），超出 K 不得分。
    - 多个方法得分相加，取得分最高的 top-K 作为最终候选。

    等分时按首次出现顺序（字典序）稳定排序。
    """
    all_sources: set = set()
    for d in candidate_dicts:
        all_sources.update(d.keys())

    merged: Dict[str, List[str]] = {}
    for src in sorted(all_sources):
        scores: Dict[str, float] = {}
        for cdict in candidate_dicts:
            for rank_0, tgt in enumerate(cdict.get(src, [])):
                score = max(K - rank_0, 0)
                if score > 0:
                    scores[tgt] = scores.get(tgt, 0.0) + score
        merged[src] = sorted(scores.keys(), key=lambda t: -scores[t])[:K]

    return merged


# ============================================================
# 评估指标
# ============================================================
def compute_metrics(
    merged: Dict[str, List[str]],
    ground_truth_file: str,
    K: int,
) -> Dict:
    """
    对比真值表计算 hit@1, hit@K, MRR。
    真值表支持列名：source_account_id / source_account / source（第 0 列兜底），
                    target_account_id / target_account / target（第 1 列兜底）。
    """
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

    hit1, hitK, mrr_sum, cnt = 0.0, 0.0, 0.0, 0
    for src, preds in merged.items():
        if src not in gt_map:
            continue
        true_tgt = gt_map[src]
        cnt += 1
        rank = next((i + 1 for i, p in enumerate(preds) if str(p) == true_tgt), None)
        if rank is not None:
            if rank == 1:
                hit1 += 1
            if rank <= K:
                hitK += 1
            mrr_sum += 1.0 / rank

    if cnt == 0:
        return {"hit@1": 0.0, f"hit@{K}": 0.0, "MRR": 0.0, "evaluated_users": 0}
    return {
        "hit@1": hit1 / cnt,
        f"hit@{K}": hitK / cnt,
        "MRR": mrr_sum / cnt,
        "evaluated_users": cnt,
    }


# ============================================================
# 打印辅助
# ============================================================
def _print_metrics_row(label: str, metrics: Dict, K: int) -> None:
    hitK_key = f"hit@{K}"
    n = metrics.get("evaluated_users", "?")
    print(
        f"  {label:<22}  "
        f"hit@1={metrics.get('hit@1', 0):.4f}  "
        f"{hitK_key}={metrics.get(hitK_key, 0):.4f}  "
        f"MRR={metrics.get('MRR', 0):.4f}  "
        f"(n={n})"
    )


# ============================================================
# 主流程
# ============================================================
def run_ensemble(
    source_file: str,
    target_file: str,
    need_align_file: str,
    ground_truth_file: str,
    output_file: str,
    K: int = 10,
    disable_gnn: bool = False,
    disable_user_picture: bool = False,
    disable_topic: bool = False,
    source_profile_file: str = "",
    target_profile_file: str = "",
    topic_device: str = "auto",
    topic_num_topics: int = 80,
) -> Tuple[str, Dict]:
    """
    并行运行三种对齐方法 + Borda Count 投票融合，输出一个 CSV 文件。

    Parameters
    ----------
    source_file          : 源平台消息 CSV 路径
    target_file          : 目标平台消息 CSV 路径
    need_align_file      : 待对齐 source 账号列表 CSV（含 account_id 列）
    ground_truth_file    : 真值对应表 CSV（用于评估）
    output_file          : 融合结果输出 CSV 路径
    K                    : Top-K 候选数量
    disable_gnn          : 跳过 GNN 方法
    disable_user_picture : 跳过 UserPicture 方法
    disable_topic        : 跳过 Topic 方法（需要 sentence_transformers + torch）
    source_profile_file  : 源平台用户画像 CSV（UserPicture 方法必需）
    target_profile_file  : 目标平台用户画像 CSV（UserPicture 方法必需）
    topic_device         : Topic embedding 设备（auto / cpu / cuda / cuda:N）
    topic_num_topics     : Topic KMeans 簇数

    Returns
    -------
    (output_file_path, ensemble_metrics)
    """
    t_total = time.time()

    print("=" * 62)
    print(f"  三路并行用户对齐  +  Borda Count 融合   K={K}")
    print("=" * 62)

    # 检查输入文件
    for fpath in (source_file, target_file, need_align_file, ground_truth_file):
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"输入文件不存在: {fpath}")

    # 各方法的临时输出文件
    tmp_dir = tempfile.mkdtemp(prefix="align_ens_")
    tmp_paths = {
        "GNN":         os.path.join(tmp_dir, "gnn_result.csv"),
        "UserPicture": os.path.join(tmp_dir, "user_picture_result.csv"),
        "Topic":       os.path.join(tmp_dir, "topic_result.csv"),
    }

    # 方法配置：(名称, loader函数, 输出路径, 额外关键字参数)
    tasks: List[Tuple[str, callable, str, dict]] = []
    if not disable_gnn:
        tasks.append(("GNN",   _get_runner_gnn,   tmp_paths["GNN"],   {}))
    if not disable_user_picture:
        if not source_profile_file or not target_profile_file:
            raise ValueError("启用 UserPicture 方法时必须指定 --source_profile_file 和 --target_profile_file")
        tasks.append(("UserPicture", _get_runner_user_picture, tmp_paths["UserPicture"],
                       {"source_profile_file": source_profile_file, "target_profile_file": target_profile_file}))
    if not disable_topic:
        tasks.append((
            "Topic",
            _get_runner_topic,
            tmp_paths["Topic"],
            {"device": topic_device, "num_topics": topic_num_topics},
        ))

    if not tasks:
        raise ValueError("至少需要启用一种对齐方法（GNN / UserPicture / Topic）。")

    print(f"\n启用方法: {[t[0] for t in tasks]}")

    method_csv: Dict[str, Optional[str]] = {}    # name -> csv_path
    method_metrics: Dict[str, Dict] = {}
    method_errors: Dict[str, str] = {}

    # ── 步骤 1：在主线程顺序加载各模块（Python import 不是线程安全的）──
    print("\n[准备] 顺序加载各方法模块（sklearn/scipy import 需串行）...")
    runners: Dict[str, callable] = {}
    active_tasks: List[Tuple[str, callable, str, dict]] = []
    for name, loader_fn, tmp, kwargs in tasks:
        try:
            runner = loader_fn()
            runners[name] = runner
            active_tasks.append((name, runner, tmp, kwargs))
            print(f"  [{name}] 模块加载完成")
        except Exception as exc:
            method_errors[name] = str(exc)
            print(f"  [{name}] 模块加载失败: {exc}")

    if not active_tasks:
        raise RuntimeError("所有方法模块均加载失败，无法继续。")

    # ── 步骤 2：并行执行各方法的实际计算（此时不再 import 新模块）──
    print(f"\n[计算] 并行运行 {[t[0] for t in active_tasks]}...\n")

    def _run_one(name: str, runner, out_tmp: str, extra_kwargs: dict):
        t0 = time.time()
        print(f"[{name}] 开始计算...")
        try:
            path, metrics = runner(
                source_file,
                target_file,
                need_align_file,
                out_tmp,
                K,
                ground_truth_file,
                **extra_kwargs,
            )
            elapsed = time.time() - t0
            print(f"[{name}] 完成 ({elapsed:.1f}s)")
            return name, path, metrics, None
        except Exception as exc:
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            print(f"[{name}] 计算失败 ({elapsed:.1f}s): {exc}")
            print(tb)
            return name, None, {}, str(exc)

    with ThreadPoolExecutor(max_workers=len(active_tasks)) as pool:
        futures = {
            pool.submit(_run_one, name, runner, tmp, kwargs): name
            for name, runner, tmp, kwargs in active_tasks
        }
        for future in as_completed(futures):
            name, path, metrics, err = future.result()
            if path is not None:
                method_csv[name] = path
                method_metrics[name] = metrics
            else:
                method_errors[name] = err

    if not method_csv:
        raise RuntimeError("所有方法均失败，无法进行 Borda Count 融合。")

    # ---------- 打印各方法评估结果 ----------
    print("\n" + "=" * 62)
    print("  各方法独立评估结果")
    print("=" * 62)
    for name in [t[0] for t in tasks]:
        if name in method_metrics and method_metrics[name]:
            _print_metrics_row(name, method_metrics[name], K)
        elif name in method_errors:
            print(f"  {name:<22}  [失败] {method_errors[name][:60]}")

    # ---------- Borda Count 融合 ----------
    print(f"\n[ensemble] 对 {len(method_csv)} 路结果执行 Borda Count 融合 (K={K})...")
    candidate_dicts = [_read_result_csv(p) for p in method_csv.values()]
    merged = borda_count_merge(candidate_dicts, K=K)

    # ---------- 写出最终结果 CSV ----------
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    out_rows = [
        {
            "source_account_id": src,
            "predict_target_account_id": ",".join(preds),
        }
        for src, preds in merged.items()
    ]
    pd.DataFrame(out_rows).to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[ensemble] 结果已写入: {output_file}  ({len(out_rows)} 行)")

    # ---------- 最终评估 ----------
    ensemble_metrics = compute_metrics(merged, ground_truth_file, K)

    print("\n" + "=" * 62)
    print("  最终融合评估结果")
    print("=" * 62)
    _print_metrics_row("Ensemble", ensemble_metrics, K)

    # 汇总对比表
    print("\n  --- 汇总对比 ---")
    print(f"  {'方法':<22}  {'hit@1':>7}  {'hit@' + str(K):>8}  {'MRR':>7}  {'n':>5}")
    print("  " + "-" * 55)
    for name in [t[0] for t in tasks]:
        if name in method_metrics and method_metrics[name]:
            m = method_metrics[name]
            print(
                f"  {name:<22}  "
                f"{m.get('hit@1', 0):>7.4f}  "
                f"{m.get(f'hit@{K}', 0):>8.4f}  "
                f"{m.get('MRR', 0):>7.4f}  "
                f"{m.get('evaluated_users', '?'):>5}"
            )
    m = ensemble_metrics
    print(
        f"  {'Ensemble (融合)':<22}  "
        f"{m.get('hit@1', 0):>7.4f}  "
        f"{m.get(f'hit@{K}', 0):>8.4f}  "
        f"{m.get('MRR', 0):>7.4f}  "
        f"{m.get('evaluated_users', '?'):>5}"
    )
    print(f"\n  总耗时: {time.time() - t_total:.1f}s")
    print("=" * 62)

    return output_file, ensemble_metrics


# ============================================================
# CLI 入口
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="三路并行用户对齐 + Borda Count 投票融合",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source_file",
        required=True,
        help="源平台消息 CSV（含 account_id, room_id, msg, time, type 列）",
    )
    parser.add_argument("--target_file", required=True, help="目标平台消息 CSV")
    parser.add_argument(
        "--need_align_file",
        required=True,
        help="待对齐 source 账号列表 CSV（含 account_id 列）",
    )
    parser.add_argument(
        "--ground_truth_file",
        required=True,
        help="真值对应表 CSV（source_account / target_account 两列）",
    )
    parser.add_argument("--output_file", required=True, help="融合结果输出 CSV 路径")
    parser.add_argument("--k", type=int, default=10, help="Top-K 候选数量")

    # UserPicture 方法参数（画像文件路径）
    parser.add_argument(
        "--source_profile_file",
        type=str,
        default="",
        help="源平台用户画像 CSV（由 picture.py 生成，UserPicture 方法必需）",
    )
    parser.add_argument(
        "--target_profile_file",
        type=str,
        default="",
        help="目标平台用户画像 CSV（由 picture.py 生成，UserPicture 方法必需）",
    )

    # 方法开关
    parser.add_argument("--disable_gnn",          action="store_true", help="跳过 GNN 方法")
    parser.add_argument("--disable_user_picture",  action="store_true", help="跳过 UserPicture 方法")
    parser.add_argument(
        "--disable_topic",
        action="store_true",
        help="跳过 Topic 方法（需要 sentence_transformers + torch，速度较慢）",
    )

    # Topic 方法额外参数
    parser.add_argument(
        "--topic_device",
        type=str,
        default="auto",
        help="Topic embedding 设备：auto / cpu / cuda / cuda:N",
    )
    parser.add_argument(
        "--topic_num_topics",
        type=int,
        default=80,
        help="Topic 方法的 KMeans 簇数",
    )

    args = parser.parse_args()

    run_ensemble(
        source_file=args.source_file,
        target_file=args.target_file,
        need_align_file=args.need_align_file,
        ground_truth_file=args.ground_truth_file,
        output_file=args.output_file,
        K=args.k,
        disable_gnn=args.disable_gnn,
        disable_user_picture=args.disable_user_picture,
        disable_topic=args.disable_topic,
        source_profile_file=args.source_profile_file,
        target_profile_file=args.target_profile_file,
        topic_device=args.topic_device,
        topic_num_topics=args.topic_num_topics,
    )


if __name__ == "__main__":
    main()
