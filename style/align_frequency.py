"""
基于发文时间/频率特征的用户对齐（可单独 import 调用）。
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from align_common import compute_topk_from_feature_dicts


def safe_divide(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """
    生成 1D 高斯核（不做归一化前后都可），用于对 24h 直方图平滑。
    使用 circular padding（小时是环）。
    """
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    if radius is None:
        radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k


def parse_time_series_to_datetime(times: pd.Series, tz_offset_hours: int = 0) -> pd.Series:
    """
    支持三类 time：
    1) unix 时间戳（int/float 或数字字符串）：自动判断秒/毫秒
    2) 'YYYY-mm-dd HH:MM:SS' 这类 datetime 字符串
    3) 无法解析的会变成 NaT
    """
    numeric = pd.to_numeric(times, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(times) > 0 else 0.0

    # 如果绝大多数都是数字，则按 unix 时间戳解析
    if numeric_ratio >= 0.8:
        median_abs = float(numeric.dropna().abs().median()) if numeric.dropna().shape[0] > 0 else 0.0
        unit = "ms" if median_abs > 1e12 else "s"
        # 对于 unix 时间戳：pandas 按 UTC 解释。若数据语义是北京时间，则需要加偏移。
        dt = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
        if tz_offset_hours:
            dt = dt + pd.Timedelta(hours=float(tz_offset_hours))
        # 返回无时区的 datetime（此时 hour 已是“偏移后的本地时钟”）
        return dt.dt.tz_localize(None)

    # 否则按 datetime 字符串解析
    # 对于 datetime 字符串：默认按原始字符串所表达的“本地时间”语义，不额外加偏移
    return pd.to_datetime(times, errors="coerce")


def frequency_24h_hist(
    df_user: pd.DataFrame,
    smooth_sigma: float = 0.0,
    alpha: float = 1.0,
    tz_offset_hours: int = 0,
) -> np.ndarray:
    """
    24 维向量：每小时发文数（归一化成比例）。
    可选：对直方图做高斯平滑，缓解小样本噪声。
    """
    times = parse_time_series_to_datetime(df_user["time"], tz_offset_hours=tz_offset_hours)
    valid_mask = times.notna()
    if valid_mask.sum() == 0:
        return np.zeros(24, dtype=float)
    hours = times[valid_mask].dt.hour.astype(int)
    counts = np.zeros(24, dtype=float)
    for h in hours.tolist():
        if 0 <= h < 24:
            counts[h] += 1
    total = counts.sum()
    if total > 0:
        counts = counts / total

    if smooth_sigma and smooth_sigma > 0:
        k = gaussian_kernel1d(float(smooth_sigma))
        radius = len(k) // 2
        # circular padding
        padded = np.concatenate([counts[-radius:], counts, counts[:radius]])
        smoothed = np.convolve(padded, k, mode="valid")
        smoothed = smoothed / smoothed.sum() if smoothed.sum() > 0 else smoothed
        counts = smoothed

    # distribution sharpening/flattening
    if alpha is not None and float(alpha) != 1.0:
        a = float(alpha)
        counts = np.power(np.maximum(counts, 0.0), a)
        s = counts.sum()
        if s > 0:
            counts = counts / s

    return counts


def frequency_coarse_bins(
    df_user: pd.DataFrame,
    night_end: int = 6,
    morning_end: int = 12,
    afternoon_end: int = 18,
    tz_offset_hours: int = 0,
) -> np.ndarray:
    times = parse_time_series_to_datetime(df_user["time"], tz_offset_hours=tz_offset_hours)
    valid_mask = times.notna()
    if valid_mask.sum() == 0:
        return np.zeros(4, dtype=float)
    hours = times[valid_mask].dt.hour.astype(int)
    bins = np.zeros(4, dtype=float)
    for h in hours.tolist():
        if 0 <= h < night_end:
            bins[0] += 1
        elif night_end <= h < morning_end:
            bins[1] += 1
        elif morning_end <= h < afternoon_end:
            bins[2] += 1
        else:
            bins[3] += 1
    total = bins.sum()
    if total > 0:
        bins = bins / total
    return bins


def frequency_peak_time_features(df_user: pd.DataFrame, tz_offset_hours: int = 0) -> tuple[float, float]:
    """
    峰值时间特征：用圆统计（circular mean/std）处理“24h是环”的情况。
    输出仍是 2 维：mean_fraction(0-1)、circ_std_fraction(0-~0.5+)
    """
    if df_user.empty:
        return 0.0, 0.0
    times = parse_time_series_to_datetime(df_user["time"], tz_offset_hours=tz_offset_hours)
    valid_mask = times.notna()
    if valid_mask.sum() == 0:
        return 0.0, 0.0
    times = times[valid_mask]

    seconds = (
        times.dt.hour.astype(float) * 3600
        + times.dt.minute.astype(float) * 60
        + times.dt.second.astype(float)
    )
    fractions = (seconds % 86400) / 86400.0

    # circular mean on angles [0, 2pi)
    angles = fractions * 2.0 * np.pi
    sin_mean = float(np.nanmean(np.sin(angles)))
    cos_mean = float(np.nanmean(np.cos(angles)))
    R = float(np.sqrt(sin_mean * sin_mean + cos_mean * cos_mean))  # 0..1
    mean_angle = float(np.arctan2(sin_mean, cos_mean))
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    mean_fraction = mean_angle / (2.0 * np.pi)

    # circular std approximation: sqrt(-2 ln(R)) / (2pi)
    if R > 0:
        circ_std = float(np.sqrt(-2.0 * np.log(R)) / (2.0 * np.pi))
    else:
        circ_std = 0.0

    return mean_fraction, circ_std


def build_frequency_features(
    df_user: pd.DataFrame,
    w24: float = 1.0,
    w_coarse: float = 1.0,
    w_peak: float = 1.0,
    smooth_sigma: float = 0.0,
    hist_alpha: float = 1.0,
    night_end: int = 6,
    morning_end: int = 12,
    afternoon_end: int = 18,
    w_dow: float = 0.5,
    normalize_groups: bool = False,
    tz_offset_hours: int = 8,
) -> np.ndarray:
    def l2_normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    v24 = (
        frequency_24h_hist(
            df_user,
            smooth_sigma=smooth_sigma,
            alpha=hist_alpha,
            tz_offset_hours=tz_offset_hours,
        )
        * float(w24)
    )
    v_coarse_raw = (
        frequency_coarse_bins(
            df_user,
            night_end=night_end,
            morning_end=morning_end,
            afternoon_end=afternoon_end,
            tz_offset_hours=tz_offset_hours,
        )
    )
    v_coarse = v_coarse_raw * float(w_coarse)
    mean_t, std_t = frequency_peak_time_features(df_user, tz_offset_hours=tz_offset_hours)

    # 额外的形状特征：峰值占比 & 熵（归一化到 [0,1]）
    # v24 已经是归一化分布，因此 max 就是峰值占比。
    peak_ratio = float(np.max(v24) / float(w24) if float(w24) != 0 else np.max(v24))
    p = np.clip(v24 / float(w24) if float(w24) != 0 else v24, 1e-12, 1.0)
    entropy = -float(np.sum(p * np.log(p)) / np.log(24.0))  # 0..1

    peak = np.array([mean_t, std_t, peak_ratio, entropy], dtype=float) * float(w_peak)

    # day-of-week histogram (0=Mon..6=Sun)
    dow = parse_time_series_to_datetime(df_user["time"], tz_offset_hours=tz_offset_hours)
    valid_mask = dow.notna()
    if valid_mask.sum() == 0:
        v_dow = np.zeros(7, dtype=float)
    else:
        dow_idx = dow[valid_mask].dt.dayofweek.astype(int)
        v_dow = np.zeros(7, dtype=float)
        for d in dow_idx.tolist():
            if 0 <= d < 7:
                v_dow[d] += 1
        s = v_dow.sum()
        if s > 0:
            v_dow = v_dow / s
    v_dow = v_dow * float(w_dow)

    if normalize_groups:
        # 逐组 L2 归一化，避免由于维度/数值尺度差异造成的主导效应。
        v24 = l2_normalize(v24)
        v_coarse = l2_normalize(v_coarse)
        peak = l2_normalize(peak)
        v_dow = l2_normalize(v_dow)

    return np.concatenate([v24, v_coarse, peak, v_dow])


def build_user_frequency_features(
    df: pd.DataFrame,
    w24: float = 1.0,
    w_coarse: float = 1.0,
    w_peak: float = 1.0,
    smooth_sigma: float = 0.0,
    hist_alpha: float = 1.0,
    night_end: int = 6,
    morning_end: int = 12,
    afternoon_end: int = 18,
    w_dow: float = 0.5,
    normalize_groups: bool = False,
    tz_offset_hours: int = 8,
) -> Dict[str, np.ndarray]:
    features: Dict[str, np.ndarray] = {}
    for account_id, df_user in df.groupby("account_id", as_index=False):
        features[str(account_id)] = build_frequency_features(
            df_user,
            w24=w24,
            w_coarse=w_coarse,
            w_peak=w_peak,
            smooth_sigma=smooth_sigma,
            hist_alpha=hist_alpha,
            night_end=night_end,
            morning_end=morning_end,
            afternoon_end=afternoon_end,
            w_dow=w_dow,
            normalize_groups=normalize_groups,
            tz_offset_hours=tz_offset_hours,
        )
    return features


def run_frequency_alignment(
    df_src: pd.DataFrame,
    df_tgt: pd.DataFrame,
    k: int,
    w24: float = 1.0,
    w_coarse: float = 1.0,
    w_peak: float = 1.0,
    smooth_sigma: float = 0.0,
    hist_alpha: float = 1.0,
    night_end: int = 6,
    morning_end: int = 12,
    afternoon_end: int = 18,
    w_dow: float = 0.5,
    normalize_groups: bool = False,
    tz_offset_hours: int = 8,
) -> pd.DataFrame:
    """
    仅用频率特征，返回含 source_account_id / predict_target_account_id 的候选表。
    """
    src_f = build_user_frequency_features(
        df_src,
        w24=w24,
        w_coarse=w_coarse,
        w_peak=w_peak,
        smooth_sigma=smooth_sigma,
        hist_alpha=hist_alpha,
        night_end=night_end,
        morning_end=morning_end,
        afternoon_end=afternoon_end,
        w_dow=w_dow,
        normalize_groups=normalize_groups,
        tz_offset_hours=tz_offset_hours,
    )
    tgt_f = build_user_frequency_features(
        df_tgt,
        w24=w24,
        w_coarse=w_coarse,
        w_peak=w_peak,
        smooth_sigma=smooth_sigma,
        hist_alpha=hist_alpha,
        night_end=night_end,
        morning_end=morning_end,
        afternoon_end=afternoon_end,
        w_dow=w_dow,
        normalize_groups=normalize_groups,
        tz_offset_hours=tz_offset_hours,
    )
    return compute_topk_from_feature_dicts(src_f, tgt_f, k=k)
