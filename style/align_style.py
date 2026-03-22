"""
基于发文风格（长度、标点、媒体类型、语气词、TF-IDF 文本）的用户对齐（可单独 import 调用）。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from align_common import compute_topk_from_feature_dicts


def safe_divide(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def style_message_length_stats(
    df_user: pd.DataFrame,
    long_len_thresh: int = 50,
    short_len_thresh: int = 10,
) -> np.ndarray:
    msgs = df_user["msg"].fillna("").astype(str)
    lengths = msgs.apply(len).values
    if len(lengths) == 0:
        return np.zeros(4, dtype=float)
    avg_len = float(lengths.mean())
    std_len = float(lengths.std())
    long_ratio = safe_divide((lengths > long_len_thresh).sum(), len(lengths))
    short_ratio = safe_divide((lengths <= short_len_thresh).sum(), len(lengths))
    return np.array([avg_len, std_len, long_ratio, short_ratio], dtype=float)


def style_punctuation_usage(
    df_user: pd.DataFrame,
    low_punct_ratio_thresh: float = 0.05,
) -> np.ndarray:
    msgs = df_user["msg"].fillna("").astype(str)
    total_chars = 0
    punct_chars = 0
    sentence_end_period = 0
    for msg in msgs:
        total_chars += len(msg)
        for ch in msg:
            if ch in "，。！？；,.!?;、":
                punct_chars += 1
        if msg.endswith(("。", ".", "！", "!", "？", "?")):
            sentence_end_period += 1
    punct_ratio = safe_divide(punct_chars, total_chars)
    period_end_ratio = safe_divide(sentence_end_period, len(msgs)) if len(msgs) > 0 else 0.0
    low_punct_flag = 1.0 if punct_ratio < float(low_punct_ratio_thresh) else 0.0
    return np.array([punct_ratio, period_end_ratio, low_punct_flag], dtype=float)


def style_emoji_photo_ratio(df_user: pd.DataFrame) -> np.ndarray:
    types = df_user["type"].fillna("").astype(str)
    n = len(types)
    if n == 0:
        return np.zeros(2, dtype=float)
    photo_ratio = safe_divide((types == "photo").sum(), n)
    video_ratio = safe_divide((types == "video").sum(), n)
    return np.array([photo_ratio, video_ratio], dtype=float)


def style_laughter_ratio(df_user: pd.DataFrame) -> np.ndarray:
    msgs = df_user["msg"].fillna("").astype(str)
    n = len(msgs)
    if n == 0:
        return np.zeros(2, dtype=float)
    laugh_count = 0
    exclaim_count = 0
    for msg in msgs:
        if "哈哈" in msg or "嘿嘿" in msg or "lol" in msg.lower():
            laugh_count += 1
        if "!" in msg or "！" in msg:
            exclaim_count += 1
    return np.array(
        [safe_divide(laugh_count, n), safe_divide(exclaim_count, n)], dtype=float
    )


def style_tfidf_features(
    msgs: List[str],
    vectorizer: TfidfVectorizer,
    agg: str = "concat",
    max_msgs: int | None = None,
) -> np.ndarray:
    """
    将用户的多条 msg 转为单个 TF-IDF 向量。

    agg:
      - concat: 将所有 msg 字符拼接后一次向量化（原实现）
      - mean: 对每条 msg 分别向量化后取均值（通常对消息条数差异更鲁棒）
    """
    if len(msgs) == 0:
        return np.zeros(len(vectorizer.vocabulary_), dtype=float)

    if max_msgs is not None and len(msgs) > max_msgs:
        msgs = msgs[:max_msgs]

    msgs = [m if isinstance(m, str) else str(m) for m in msgs]

    if agg == "concat":
        tfidf = vectorizer.transform(["".join(msgs)])
        vec = tfidf.toarray().reshape(-1)
    elif agg == "mean":
        tfidf = vectorizer.transform(msgs)
        vec = tfidf.toarray().mean(axis=0).reshape(-1)
    else:
        raise ValueError(f"Unknown agg: {agg}")

    if np.linalg.norm(vec) > 0:
        vec = vec / np.linalg.norm(vec)
    return vec


def build_style_features_without_text(
    df_user: pd.DataFrame,
    long_len_thresh: int = 50,
    short_len_thresh: int = 10,
    low_punct_ratio_thresh: float = 0.05,
) -> np.ndarray:
    return np.concatenate(
        [
            style_message_length_stats(
                df_user,
                long_len_thresh=long_len_thresh,
                short_len_thresh=short_len_thresh,
            ),
            style_punctuation_usage(df_user, low_punct_ratio_thresh=low_punct_ratio_thresh),
            style_emoji_photo_ratio(df_user),
            style_laughter_ratio(df_user),
        ]
    )


def build_global_tfidf(
    df_all: pd.DataFrame,
    max_features: int = 500,
    ngram_min: int = 2,
    ngram_max: int = 4,
    min_df: int | float = 1,
    sublinear_tf: bool = False,
) -> TfidfVectorizer:
    corpus = df_all["msg"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    vectorizer.fit(corpus)
    return vectorizer


def build_user_style_features(
    df: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer,
    use_tfidf: bool = True,
    basic_weight: float = 1.0,
    tfidf_weight: float = 1.0,
    long_len_thresh: int = 50,
    short_len_thresh: int = 10,
    low_punct_ratio_thresh: float = 0.05,
    tfidf_agg: str = "concat",
    tfidf_max_msgs: int | None = None,
) -> Dict[str, np.ndarray]:
    features: Dict[str, np.ndarray] = {}
    for account_id, df_user in df.groupby("account_id", as_index=False):
        basic = (
            build_style_features_without_text(
                df_user,
                long_len_thresh=long_len_thresh,
                short_len_thresh=short_len_thresh,
                low_punct_ratio_thresh=low_punct_ratio_thresh,
            )
            * float(basic_weight)
        )
        if use_tfidf and tfidf_vectorizer is not None:
            msgs = df_user["msg"].fillna("").astype(str).tolist()
            text = style_tfidf_features(
                msgs,
                tfidf_vectorizer,
                agg=tfidf_agg,
                max_msgs=tfidf_max_msgs,
            )
            vec = np.concatenate([basic, text * float(tfidf_weight)])
        else:
            vec = basic
        features[str(account_id)] = vec
    return features


def run_style_alignment(
    df_src: pd.DataFrame,
    df_tgt: pd.DataFrame,
    k: int,
    max_tfidf_features: int = 500,
    use_tfidf: bool = True,
    basic_weight: float = 1.0,
    tfidf_weight: float = 1.0,
    long_len_thresh: int = 50,
    short_len_thresh: int = 10,
    low_punct_ratio_thresh: float = 0.05,
    tfidf_ngram_min: int = 2,
    tfidf_ngram_max: int = 4,
    tfidf_min_df: int | float = 1,
    tfidf_sublinear_tf: bool = False,
    tfidf_agg: str = "concat",
    tfidf_max_msgs: int | None = None,
) -> pd.DataFrame:
    """
    风格特征 + 可选 TF-IDF，返回候选表。
    """
    if use_tfidf:
        df_all = pd.concat([df_src, df_tgt], axis=0, ignore_index=True)
        tfidf = build_global_tfidf(
            df_all,
            max_features=max_tfidf_features,
            ngram_min=tfidf_ngram_min,
            ngram_max=tfidf_ngram_max,
            min_df=tfidf_min_df,
            sublinear_tf=tfidf_sublinear_tf,
        )
    else:
        tfidf = None
    src_f = build_user_style_features(
        df_src,
        tfidf,
        use_tfidf=use_tfidf,
        basic_weight=basic_weight,
        tfidf_weight=tfidf_weight,
        long_len_thresh=long_len_thresh,
        short_len_thresh=short_len_thresh,
        low_punct_ratio_thresh=low_punct_ratio_thresh,
        tfidf_agg=tfidf_agg,
        tfidf_max_msgs=tfidf_max_msgs,
    )
    tgt_f = build_user_style_features(
        df_tgt,
        tfidf,
        use_tfidf=use_tfidf,
        basic_weight=basic_weight,
        tfidf_weight=tfidf_weight,
        long_len_thresh=long_len_thresh,
        short_len_thresh=short_len_thresh,
        low_punct_ratio_thresh=low_punct_ratio_thresh,
        tfidf_agg=tfidf_agg,
        tfidf_max_msgs=tfidf_max_msgs,
    )
    return compute_topk_from_feature_dicts(src_f, tgt_f, k=k)
