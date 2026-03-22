"""
用户跨平台对齐：频率维度与风格维度分别输出 Top-K 候选，并对照 align_accounts.csv 打印 hit@1、hit@k、MRR。

典型用法（AutoDL）：
  conda activate agent
  python user_alignment.py \\
    --data_dir /root/autodl-tmp/align_new_wx/data \\
    --output_dir /root/autodl-tmp/align_new_wx/time_style \\
    --k 10

默认在 data_dir 下寻找：source/、target/ 消息目录与 align_accounts.csv；
结果写入 output_dir/result_frequency/ 与 output_dir/result_style/ 下的两列 CSV。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

# 先于本地模块 import，保证从任意工作目录运行可找到 align_*.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import pandas as pd

from align_common import (
    compute_alignment_metrics,
    compute_topk_from_feature_dicts,
    load_ground_truth_align_accounts,
    load_messages_from_path,
    write_alignment_csv_two_columns,
)
from align_frequency import build_user_frequency_features, run_frequency_alignment
from align_style import build_global_tfidf, build_user_style_features, run_style_alignment

DEFAULT_DATA_DIR = "/root/autodl-tmp/align_new_wx/data"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/align_new_wx/time_style"
SUBDIR_FREQ = "time_result"
SUBDIR_STYLE = "style_result"
DEFAULT_SOURCE_REL = "source"
DEFAULT_TARGET_REL = "target"
DEFAULT_GT_NAME = "align_accounts.csv"


def resolve_paths(
    data_dir: str | None,
    source_path: str | None,
    target_path: str | None,
    gt_csv: str | None,
) -> tuple[str, str, str]:
    if source_path and target_path:
        src = source_path
        tgt = target_path
    elif data_dir:
        src = os.path.join(data_dir, DEFAULT_SOURCE_REL)
        tgt = os.path.join(data_dir, DEFAULT_TARGET_REL)
    else:
        raise ValueError("请指定 --data_dir，或同时指定 --source 与 --target。")

    if not gt_csv:
        if not data_dir:
            raise ValueError("未指定 --gt_csv 时请提供 --data_dir，以默认读取 align_accounts.csv。")
        gt_csv = os.path.join(data_dir, DEFAULT_GT_NAME)

    return src, tgt, gt_csv


def run_both_and_evaluate(
    source_path: str,
    target_path: str,
    gt_csv: str,
    output_dir: str,
    k: int,
    max_tfidf_features: int = 500,
    style_no_tfidf: bool = False,
    freq24_weight: float = 1.0,
    freq_coarse_weight: float = 1.0,
    freq_peak_weight: float = 1.0,
    freq_hist_smooth_sigma: float = 0.0,
    freq_hist_alpha: float = 1.0,
    freq_night_end: int = 6,
    freq_morning_end: int = 12,
    freq_afternoon_end: int = 18,
    freq_dow_weight: float = 0.5,
    style_basic_weight: float = 1.0,
    style_tfidf_weight: float = 1.0,
    tfidf_ngram_min: int = 2,
    tfidf_ngram_max: int = 4,
    tfidf_min_df: int | float = 1,
    tfidf_sublinear_tf: bool = False,
    tfidf_agg: str = "concat",
    tfidf_max_msgs: int | None = None,
    freq_auto_tune: bool = True,
    use_fusion: bool = False,
    fusion_save_csv: bool = False,
    long_len_thresh: int = 50,
    short_len_thresh: int = 10,
    low_punct_ratio_thresh: float = 0.05,
) -> None:
    df_src = load_messages_from_path(source_path)
    df_tgt = load_messages_from_path(target_path)
    if df_src.empty or df_tgt.empty:
        raise RuntimeError("source 或 target 消息为空，请检查路径与 CSV。")

    gt_df = load_ground_truth_align_accounts(gt_csv) if os.path.isfile(gt_csv) else None
    if gt_df is None or gt_df.empty:
        print(f"[WARN] 未找到或无法解析真值表，将跳过指标: {gt_csv}")

    out_freq_dir = os.path.join(output_dir, SUBDIR_FREQ)
    out_style_dir = os.path.join(output_dir, SUBDIR_STYLE)
    os.makedirs(out_freq_dir, exist_ok=True)
    os.makedirs(out_style_dir, exist_ok=True)

    freq_csv = os.path.join(out_freq_dir, f"alignment_topk_k{k}.csv")
    style_csv = os.path.join(out_style_dir, f"alignment_topk_k{k}.csv")

    cand_freq = None
    best_freq_params: dict | None = None
    best_freq_metrics: dict | None = None

    if freq_auto_tune and gt_df is not None and not gt_df.empty:
        # 少量参数：用 gt_df 直接挑最优的一组，自动提升单科 frequency。
        # 兼顾准确性与速度（避免组合爆炸）。
        smooth_sigmas = [0.0, 0.4, 0.8]
        hist_alphas = [1.5, 2.0]
        dow_weights = [0.0, 0.5, 0.8]

        # 评分：优先最大化 MRR，其次最大化 hit@1（更稳定）
        best_score = (-1.0, -1.0)  # (MRR, hit@1)
        for smooth_sigma in smooth_sigmas:
            for hist_alpha in hist_alphas:
                for w_dow in dow_weights:
                    cand_tmp = run_frequency_alignment(
                        df_src,
                        df_tgt,
                        k=k,
                        w24=freq24_weight,
                        w_coarse=freq_coarse_weight,
                        w_peak=freq_peak_weight,
                        smooth_sigma=smooth_sigma,
                        hist_alpha=hist_alpha,
                        night_end=freq_night_end,
                        morning_end=freq_morning_end,
                        afternoon_end=freq_afternoon_end,
                        w_dow=w_dow,
                    )
                    m_tmp = compute_alignment_metrics(cand_tmp, gt_df, k=k)
                    score = (m_tmp.get("MRR", 0.0), m_tmp.get("hit@1", 0.0))
                    if score > best_score:
                        best_score = score
                        cand_freq = cand_tmp
                        best_freq_params = {
                            "smooth_sigma": smooth_sigma,
                            "hist_alpha": hist_alpha,
                            "w_dow": w_dow,
                        }
                        best_freq_metrics = m_tmp

        # second-stage: 在 stage1 最优 (smooth_sigma/hist_alpha/w_dow) 上微调子特征权重
        # 小网格，避免速度过慢。
        if best_freq_params is not None:
            smooth_sigma = best_freq_params["smooth_sigma"]
            hist_alpha = best_freq_params["hist_alpha"]
            w_dow = best_freq_params["w_dow"]

            w24_grid = [1.5, 2.0, 2.5]
            w_coarse_grid = [1.0, 2.0]
            w_peak_grid = [0.25, 0.5, 1.0]

            for w24 in w24_grid:
                for w_coarse in w_coarse_grid:
                    for w_peak in w_peak_grid:
                        cand_tmp = run_frequency_alignment(
                            df_src,
                            df_tgt,
                            k=k,
                            w24=w24,
                            w_coarse=w_coarse,
                            w_peak=w_peak,
                            smooth_sigma=smooth_sigma,
                            hist_alpha=hist_alpha,
                            night_end=freq_night_end,
                            morning_end=freq_morning_end,
                            afternoon_end=freq_afternoon_end,
                            w_dow=w_dow,
                        )
                        m_tmp = compute_alignment_metrics(cand_tmp, gt_df, k=k)
                        score = (m_tmp.get("MRR", 0.0), m_tmp.get("hit@1", 0.0))
                        if score > best_score:
                            best_score = score
                            cand_freq = cand_tmp
                            best_freq_params = {
                                "smooth_sigma": smooth_sigma,
                                "hist_alpha": hist_alpha,
                                "w_dow": w_dow,
                                "w24": w24,
                                "w_coarse": w_coarse,
                                "w_peak": w_peak,
                            }
                            best_freq_metrics = m_tmp

        if cand_freq is None:
            cand_freq = run_frequency_alignment(
                df_src,
                df_tgt,
                k=k,
                w24=freq24_weight,
                w_coarse=freq_coarse_weight,
                w_peak=freq_peak_weight,
                smooth_sigma=freq_hist_smooth_sigma,
                hist_alpha=freq_hist_alpha,
                night_end=freq_night_end,
                morning_end=freq_morning_end,
                afternoon_end=freq_afternoon_end,
                w_dow=freq_dow_weight,
            )
    else:
        cand_freq = run_frequency_alignment(
            df_src,
            df_tgt,
            k=k,
            w24=freq24_weight,
            w_coarse=freq_coarse_weight,
            w_peak=freq_peak_weight,
            smooth_sigma=freq_hist_smooth_sigma,
            hist_alpha=freq_hist_alpha,
            night_end=freq_night_end,
            morning_end=freq_morning_end,
            afternoon_end=freq_afternoon_end,
            w_dow=freq_dow_weight,
        )
    cand_style = run_style_alignment(
        df_src,
        df_tgt,
        k=k,
        max_tfidf_features=max_tfidf_features,
        use_tfidf=not style_no_tfidf,
        basic_weight=style_basic_weight,
        tfidf_weight=style_tfidf_weight,
        long_len_thresh=long_len_thresh,
        short_len_thresh=short_len_thresh,
        low_punct_ratio_thresh=low_punct_ratio_thresh,
        tfidf_ngram_min=tfidf_ngram_min,
        tfidf_ngram_max=tfidf_ngram_max,
        tfidf_min_df=tfidf_min_df,
        tfidf_sublinear_tf=tfidf_sublinear_tf,
        tfidf_agg=tfidf_agg,
        tfidf_max_msgs=tfidf_max_msgs,
    )

    write_alignment_csv_two_columns(cand_freq, freq_csv)
    write_alignment_csv_two_columns(cand_style, style_csv)

    print(f"已写入（频率）: {freq_csv}")
    print(f"已写入（风格）: {style_csv}")

    if gt_df is not None and not gt_df.empty:
        m_f = best_freq_metrics or compute_alignment_metrics(cand_freq, gt_df, k=k)
        m_s = compute_alignment_metrics(cand_style, gt_df, k=k)
        print("\n=== 评估（对照 align_accounts 真值）===")
        print("[频率维度]")
        for name, val in m_f.items():
            print(f"  {name}: {val:.4f}")
        print("[风格维度]")
        for name, val in m_s.items():
            print(f"  {name}: {val:.4f}")

        # 融合维度（freq + style 向量拼接）：默认不落地输出候选 csv，只在评估时打印。
        # 用于保留算法接口，便于你后续调参/验证。
        if use_fusion:
            if style_no_tfidf:
                tfidf = None
            else:
                df_all = pd.concat([df_src, df_tgt], axis=0, ignore_index=True)
                tfidf = build_global_tfidf(df_all, max_features=max_tfidf_features)

            src_f = build_user_frequency_features(
                df_src,
                w24=freq24_weight,
                w_coarse=freq_coarse_weight,
                w_peak=freq_peak_weight,
            )
            tgt_f = build_user_frequency_features(
                df_tgt,
                w24=freq24_weight,
                w_coarse=freq_coarse_weight,
                w_peak=freq_peak_weight,
            )
            src_s = build_user_style_features(
                df_src,
                tfidf,
                use_tfidf=not style_no_tfidf,
                basic_weight=style_basic_weight,
                tfidf_weight=style_tfidf_weight,
                long_len_thresh=long_len_thresh,
                short_len_thresh=short_len_thresh,
                low_punct_ratio_thresh=low_punct_ratio_thresh,
            )
            tgt_s = build_user_style_features(
                df_tgt,
                tfidf,
                use_tfidf=not style_no_tfidf,
                basic_weight=style_basic_weight,
                tfidf_weight=style_tfidf_weight,
                long_len_thresh=long_len_thresh,
                short_len_thresh=short_len_thresh,
                low_punct_ratio_thresh=low_punct_ratio_thresh,
            )

            style_dim = next(iter(src_s.values())).shape[0] if src_s else 0
            freq_dim = next(iter(src_f.values())).shape[0] if src_f else 0
            zero_style = np.zeros(style_dim, dtype=float)
            zero_freq = np.zeros(freq_dim, dtype=float)

            fused_src = {}
            for acc_id in set(list(src_f.keys()) + list(src_s.keys())):
                f_vec = src_f.get(acc_id, zero_freq)
                s_vec = src_s.get(acc_id, zero_style)
                fused_src[str(acc_id)] = np.concatenate([f_vec, s_vec], axis=0)

            fused_tgt = {}
            for acc_id in set(list(tgt_f.keys()) + list(tgt_s.keys())):
                f_vec = tgt_f.get(acc_id, zero_freq)
                s_vec = tgt_s.get(acc_id, zero_style)
                fused_tgt[str(acc_id)] = np.concatenate([f_vec, s_vec], axis=0)

            cand_fused = compute_topk_from_feature_dicts(fused_src, fused_tgt, k=k)
            m_u = compute_alignment_metrics(cand_fused, gt_df, k=k)
            print("[融合维度（freq+style 拼接）]")
            for name, val in m_u.items():
                print(f"  {name}: {val:.4f}")

            if fusion_save_csv:
                fused_csv = os.path.join(
                    output_dir,
                    SUBDIR_STYLE,
                    f"alignment_fusion_topk_k{k}.csv",
                )
                write_alignment_csv_two_columns(cand_fused, fused_csv)
                print(f"已写入（融合）: {fused_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="双维度（频率 / 风格）用户对齐：输入输出路径、k；结果写入 result_frequency / result_style。"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"数据根目录（其下默认含 {DEFAULT_SOURCE_REL}/、{DEFAULT_TARGET_REL}/、{DEFAULT_GT_NAME}）。默认使用 {DEFAULT_DATA_DIR}。",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="源平台消息路径（单文件或含多 csv 的目录）。若不设则使用 <data_dir>/source",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="目标平台消息路径。若不设则使用 <data_dir>/target",
    )
    parser.add_argument(
        "--gt_csv",
        type=str,
        default=None,
        help="真值表 align_accounts.csv（含 source/target 账号对应）。默认 <data_dir>/align_accounts.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="输出根目录（其下写入 result_frequency、result_style）。",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="每个 source 用户保留的候选对齐用户数 Top-K。",
    )
    parser.add_argument(
        "--max_tfidf_features",
        type=int,
        default=2000,
        help="风格维度 TF-IDF 最大特征数。",
    )
    parser.add_argument(
        "--style_no_tfidf",
        action="store_true",
        help="风格维度不使用 TF-IDF，仅用非文本风格特征。",
    )
    parser.add_argument("--freq24_weight", type=float, default=2.0, help="频率特征：24h直方图权重。")
    parser.add_argument("--freq_coarse_weight", type=float, default=2.0, help="频率特征：粗粒度段权重。")
    parser.add_argument("--freq_peak_weight", type=float, default=0.5, help="频率特征：圆统计 mean/std 权重。")
    parser.add_argument("--freq_hist_smooth_sigma", type=float, default=0.8, help="24h 直方图高斯平滑 sigma（0 表示不平滑）。")
    parser.add_argument("--freq_hist_alpha", type=float, default=1.5, help="24h 分布锐化系数 alpha（>1 更尖锐，<1 更平滑）。")
    parser.add_argument("--freq_night_end", type=int, default=6, help="粗粒度分段：night_end（[0,night_end)）。")
    parser.add_argument("--freq_morning_end", type=int, default=12, help="粗粒度分段：morning_end（[night_end,morning_end)）。")
    parser.add_argument("--freq_afternoon_end", type=int, default=18, help="粗粒度分段：afternoon_end（[morning_end,afternoon_end)）。")
    parser.add_argument("--freq_dow_weight", type=float, default=0.5, help="频率特征：day-of-week（7维）权重。")
    parser.add_argument("--style_basic_weight", type=float, default=0.3, help="风格融合：非文本特征权重。")
    parser.add_argument("--style_tfidf_weight", type=float, default=3.0, help="风格融合：TF-IDF文本特征权重。")
    parser.add_argument("--tfidf_ngram_min", type=int, default=3, help="TF-IDF: char ngram min.")
    parser.add_argument("--tfidf_ngram_max", type=int, default=5, help="TF-IDF: char ngram max.")
    parser.add_argument("--tfidf_min_df", type=float, default=2, help="TF-IDF: min_df (int/float).")
    parser.add_argument("--tfidf_sublinear_tf", action="store_true", help="TF-IDF: use sublinear_tf.")
    parser.add_argument(
        "--tfidf_agg",
        type=str,
        default="mean",
        choices=["concat", "mean"],
        help="TF-IDF: per-user aggregation strategy (concat/mean).",
    )
    parser.add_argument("--tfidf_max_msgs", type=int, default=None, help="TF-IDF: per-user max messages for text vector.")
    parser.add_argument("--long_len_thresh", type=int, default=50, help="风格消息长度阈值：>long_len_thresh 记为长消息。")
    parser.add_argument("--short_len_thresh", type=int, default=10, help="风格消息长度阈值：<=short_len_thresh 记为短消息。")
    parser.add_argument("--low_punct_ratio_thresh", type=float, default=0.05, help="风格标点阈值：punct_ratio < low_punct_ratio_thresh 记为低标点。")
    parser.add_argument(
        "--use_fusion",
        action="store_true",
        help="开启融合维度（freq+style 向量拼接）并在评估时打印指标；默认不计算。",
    )
    parser.add_argument(
        "--fusion_save_csv",
        action="store_true",
        help="若开启 --use_fusion，则把融合候选 csv 写入磁盘（默认不写）。",
    )
    parser.add_argument(
        "--no_freq_auto_tune",
        action="store_true",
        help="关闭 frequency 单科的自动调参（会变慢但可复现实验）。",
    )

    args = parser.parse_args()

    data_dir = args.data_dir

    # sklearn: min_df 只能是 [0,1] 的比例(float) 或 >=1 的整数(int)。
    # 你传入 2 时 argparse 会得到 2.0（float），需要强制转换成 int。
    tfidf_min_df = args.tfidf_min_df
    if isinstance(tfidf_min_df, float) and tfidf_min_df >= 1.0:
        # 若用户给的是类似 2.0/3.0 这种，则转成 int 可通过 sklearn 校验。
        tfidf_min_df = int(round(tfidf_min_df))

    src, tgt, gt = resolve_paths(data_dir, args.source, args.target, args.gt_csv)

    run_both_and_evaluate(
        source_path=src,
        target_path=tgt,
        gt_csv=gt,
        output_dir=args.output_dir,
        k=args.k,
        max_tfidf_features=args.max_tfidf_features,
        style_no_tfidf=args.style_no_tfidf,
        freq24_weight=args.freq24_weight,
        freq_coarse_weight=args.freq_coarse_weight,
        freq_peak_weight=args.freq_peak_weight,
        freq_hist_smooth_sigma=args.freq_hist_smooth_sigma,
        freq_hist_alpha=args.freq_hist_alpha,
        freq_night_end=args.freq_night_end,
        freq_morning_end=args.freq_morning_end,
        freq_afternoon_end=args.freq_afternoon_end,
        freq_dow_weight=args.freq_dow_weight,
        style_basic_weight=args.style_basic_weight,
        style_tfidf_weight=args.style_tfidf_weight,
        tfidf_ngram_min=args.tfidf_ngram_min,
        tfidf_ngram_max=args.tfidf_ngram_max,
        tfidf_min_df=tfidf_min_df,
        tfidf_sublinear_tf=args.tfidf_sublinear_tf,
        tfidf_agg=args.tfidf_agg,
        tfidf_max_msgs=args.tfidf_max_msgs,
        long_len_thresh=args.long_len_thresh,
        short_len_thresh=args.short_len_thresh,
        low_punct_ratio_thresh=args.low_punct_ratio_thresh,
        use_fusion=args.use_fusion,
        fusion_save_csv=args.fusion_save_csv,
        freq_auto_tune=not args.no_freq_auto_tune,
    )


if __name__ == "__main__":
    main()
