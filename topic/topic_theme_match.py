import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

# 输入 CSV 默认目录（可用各 --*_csv 参数覆盖为任意路径）
_DEFAULT_INPUT_DATA_DIR = "/root/autodl-tmp/align_new_wx/data"
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm


def _unix_day_id(unix_ts: np.ndarray, tz_offset_hours: int) -> np.ndarray:
    # unix_ts is seconds. "day id" is an integer bucket.
    seconds_offset = int(tz_offset_hours) * 3600
    return ((unix_ts.astype(np.int64) + seconds_offset) // 86400).astype(np.int64)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def _load_need_align_accounts(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "account_id" not in df.columns:
        raise ValueError(f"Expected column `account_id` in {path}, got {df.columns.tolist()}")
    return df["account_id"].astype(str).tolist()


def _parse_account_index(account_id: str) -> int:
    # Expected: s_user_001 / t_user_001
    # Keep it robust for varying zero padding.
    parts = account_id.split("_")
    # last part should be the numeric index
    return int(parts[-1])


def _predict_target_id_from_source(source_account_id: str) -> str:
    idx = _parse_account_index(source_account_id)
    return f"t_user_{idx:03d}"


def _maybe_login_hf():
    # Don't print tokens; just enable private/gated downloads if user already set HF_TOKEN.
    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token)
    except Exception:
        # If login fails, we'll rely on SentenceTransformer/Transformers auth behavior.
        pass


def _resolve_sentence_transformer_device(device: str) -> Tuple[str, str]:
    """
    将 CLI 的 device 解析为 SentenceTransformer 可用的设备字符串，并返回简短说明用于日志。
    """
    d = str(device).strip().lower()
    try:
        import torch
    except ImportError as e:
        raise ImportError("需要安装 PyTorch 才能使用 GPU/CPU embedding，请先安装 torch。") from e

    if d in ("auto", ""):
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return f"cuda:{idx}", f"CUDA:{idx} ({name})"
        return "cpu", "cpu (未检测到 CUDA，使用 CPU)"

    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前 PyTorch 未检测到 CUDA（驱动/CUDA 版本不匹配或未装 GPU 版 torch）。")
        idx = 0
        name = torch.cuda.get_device_name(idx)
        return "cuda:0", f"cuda:0 ({name})"

    if d.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"指定了 --device {device}，但当前 PyTorch 未检测到 CUDA。")
        try:
            idx = int(d.split(":", 1)[1])
        except ValueError as e:
            raise ValueError(f"无效的 --device 值: {device}") from e
        if idx < 0 or idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"--device {device} 无效：当前仅有 {torch.cuda.device_count()} 块 GPU（索引 0..{torch.cuda.device_count()-1}）。"
            )
        name = torch.cuda.get_device_name(idx)
        return f"cuda:{idx}", f"cuda:{idx} ({name})"

    if d == "cpu":
        return "cpu", "cpu"

    raise ValueError(f"不支持的 --device: {device}，请使用 auto / cpu / cuda / cuda:N")


def embed_texts(
    model_name: str,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: str = "auto",
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    _maybe_login_hf()
    st_device, st_device_desc = _resolve_sentence_transformer_device(device)
    print(f"[Embed] SentenceTransformer device: {st_device_desc}")
    model = SentenceTransformer(model_name, device=st_device)

    # sentence-transformers 的不同版本/不同模型对 max_length 支持不一致：
    # 有些模型接受 encode(..., max_length=...), 有些不接受。这里统一采用“尽量设置模型最大序列长度”的方式。
    if hasattr(model, "max_seq_length"):
        try:
            model.max_seq_length = int(max_length)
        except Exception:
            pass

    # sentence-transformers returns float32 numpy arrays by default.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # We'll control normalization ourselves.
    )
    return embeddings.astype(np.float32)


def build_user_day_topic_hist(
    df_text: pd.DataFrame,
    topic_labels: np.ndarray,
    num_topics: int,
) -> Dict[str, List[Tuple[int, np.ndarray]]]:
    """
    Returns:
      themes_by_account[account_id] = [(day_id, L2_normalized_topic_distribution_vector), ...]
    """
    tmp = df_text[["account_id", "day_id", "type"]].copy()  # type is not used, but keep schema explicit
    tmp = tmp.drop(columns=["type"])
    tmp["topic"] = topic_labels

    # Count topic occurrences per (account_id, day_id, topic).
    g = tmp.groupby(["account_id", "day_id", "topic"]).size().reset_index(name="count")

    # Pivot to (account_id, day_id) x topic matrix.
    pivot = g.pivot_table(
        index=["account_id", "day_id"],
        columns="topic",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )

    # Ensure all topic columns exist (KMeans always gives topics in [0, num_topics)).
    # pivot columns can be sparse depending on data.
    if pivot.shape[1] != num_topics:
        for t in range(num_topics):
            if t not in pivot.columns:
                pivot[t] = 0.0
        pivot = pivot.sort_index(axis=1)

    H = pivot.values.astype(np.float32)

    # Topic distribution (sum to 1) then L2 normalize for cosine similarity.
    H_sum = H.sum(axis=1, keepdims=True)
    H_dist = H / np.clip(H_sum, 1e-12, None)
    H_norm = _l2_normalize(H_dist)

    themes_by_account: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    for (account_id, day_id), vec in zip(pivot.index, H_norm):
        themes_by_account.setdefault(str(account_id), []).append((int(day_id), vec))

    # Sort day buckets for stable behavior.
    for acc in list(themes_by_account.keys()):
        themes_by_account[acc].sort(key=lambda x: x[0])

    return themes_by_account


def build_user_day_topic_hist_soft(
    df_text: pd.DataFrame,
    topic_probs: np.ndarray,
    num_topics: int,
) -> Dict[str, List[Tuple[int, np.ndarray]]]:
    """
    Soft version:
      - 每条发言 -> 对各个 topic 的概率分布
      - 按 (account_id, day_id) 聚合概率求和 -> 当天用户的 topic 分布向量
    """
    if topic_probs.shape[1] != num_topics:
        raise ValueError(f"topic_probs second dim must be {num_topics}, got {topic_probs.shape[1]}")

    tmp = df_text[["account_id", "day_id"]].copy()

    # 使用不易冲突的分隔符做 pair-key，方便从 uniques 中还原 account_id/day_id
    tmp["pair_key"] = tmp["account_id"].astype(str) + "\t" + tmp["day_id"].astype(str)
    pair_ids, pair_keys = pd.factorize(tmp["pair_key"], sort=False)

    g = len(pair_keys)
    H = np.zeros((g, num_topics), dtype=np.float32)
    np.add.at(H, pair_ids, topic_probs.astype(np.float32))

    # 概率分布化：sum=1，再做 L2 normalize（cosine similarity 更稳）
    H_sum = H.sum(axis=1, keepdims=True)
    H_dist = H / np.clip(H_sum, 1e-12, None)
    H_norm = _l2_normalize(H_dist)

    themes_by_account: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    for i, key in enumerate(pair_keys):
        acc_id, day_s = str(key).split("\t", 1)
        day_id = int(day_s)
        themes_by_account.setdefault(str(acc_id), []).append((day_id, H_norm[i]))

    # Sort day buckets for stable behavior.
    for acc in list(themes_by_account.keys()):
        themes_by_account[acc].sort(key=lambda x: x[0])

    return themes_by_account


def score_source_against_targets(
    src_days: np.ndarray,
    src_vecs: np.ndarray,
    tgt_days: np.ndarray,
    tgt_vecs: np.ndarray,
    time_window_days: int,
) -> float:
    """
    Score between two accounts using:
      score = mean_over_source_days( max_cos_sim(target_day_vec, src_day_vec) within window )
    Cosine sim computed via dot product since vectors are L2-normalized.
    """
    # src_days: (Ds,), src_vecs: (Ds, T)
    # tgt_days: (Dt,), tgt_vecs: (Dt, T)
    total = 0.0
    valid_source_day_cnt = 0

    for day_idx in range(src_vecs.shape[0]):
        day_s = src_days[day_idx]
        # 仅允许 target 在 source 之后：
        #   src_day <= tgt_day <= src_day + time_window_days
        mask = (tgt_days >= day_s) & (tgt_days <= (day_s + time_window_days))
        if not np.any(mask):
            continue  # no compatible target day bucket; ignore this source day
        sims = tgt_vecs[mask] @ src_vecs[day_idx]  # (num_matched_days,)
        total += float(np.max(sims))
        valid_source_day_cnt += 1

    if valid_source_day_cnt == 0:
        return 0.0
    return total / valid_source_day_cnt


def build_user_day_embed_avg(
    df_text: pd.DataFrame,
    text_emb: np.ndarray,
) -> Dict[str, List[Tuple[int, np.ndarray]]]:
    """
    对每个 (account_id, day_id) 计算文本 embedding 的均值并 L2 normalize。
    """
    if len(df_text) != text_emb.shape[0]:
        raise ValueError("df_text and text_emb row count mismatch.")

    dim = text_emb.shape[1]
    tmp = df_text[["account_id", "day_id"]].copy()
    tmp["pair_key"] = tmp["account_id"].astype(str) + "\t" + tmp["day_id"].astype(str)
    pair_ids, pair_keys = pd.factorize(tmp["pair_key"], sort=False)

    g = len(pair_keys)
    S = np.zeros((g, dim), dtype=np.float32)
    np.add.at(S, pair_ids, text_emb.astype(np.float32))
    counts = np.bincount(pair_ids).astype(np.float32)
    avg = S / np.clip(counts[:, None], 1e-12, None)
    avg_norm = _l2_normalize(avg)

    by_account: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    for i, key in enumerate(pair_keys):
        acc_id, day_s = str(key).split("\t", 1)
        by_account.setdefault(acc_id, []).append((int(day_s), avg_norm[i]))

    for acc in list(by_account.keys()):
        by_account[acc].sort(key=lambda x: x[0])
    return by_account


def build_user_global_embed_avg(
    df_text: pd.DataFrame,
    text_emb: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    每个用户全局 embedding 均值（对该用户所有 text 发言聚合），并 L2 normalize。
    """
    if len(df_text) != text_emb.shape[0]:
        raise ValueError("df_text and text_emb row count mismatch.")

    dim = text_emb.shape[1]
    # factorize account_id -> row groups
    acc_ids = df_text["account_id"].astype(str).values
    unique_acc, inv = np.unique(acc_ids, return_inverse=True)
    g = len(unique_acc)

    S = np.zeros((g, dim), dtype=np.float32)
    np.add.at(S, inv, text_emb.astype(np.float32))
    counts = np.bincount(inv).astype(np.float32)
    avg = S / np.clip(counts[:, None], 1e-12, None)
    avg_norm = _l2_normalize(avg)

    return {str(unique_acc[i]): avg_norm[i] for i in range(g)}


def compute_metrics(
    scores_sorted_targets: List[str],
    gt_target_account_id: str,
    topk: int,
) -> Tuple[int, int, float]:
    """
    Returns (hit1, hitK, mrr)
    - hit1: gt at rank 1
    - hitK: gt within topK
    - mrr: 1/rank if gt is present else 0
    """
    try:
        rank0 = scores_sorted_targets.index(gt_target_account_id)
    except ValueError:
        return 0, 0, 0.0

    rank = rank0 + 1
    hit1 = 1 if rank == 1 else 0
    hitK = 1 if rank <= topk else 0
    mrr = 1.0 / rank
    return hit1, hitK, mrr


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    x = x / max(float(temperature), 1e-12)
    x = x - np.max(x, axis=1, keepdims=True)  # numerical stability
    exp_x = np.exp(x, dtype=np.float32)
    return exp_x / np.clip(exp_x.sum(axis=1, keepdims=True), 1e-12, None)


def _parse_csv_ints(csv_text: str) -> List[int]:
    return [int(x.strip()) for x in str(csv_text).split(",") if x.strip()]


def _parse_csv_floats(csv_text: str) -> List[float]:
    return [float(x.strip()) for x in str(csv_text).split(",") if x.strip()]


def _fit_topic_cluster_model(
    X_fit: np.ndarray,
    n_clusters: int,
    *,
    cluster_backend: str,
    random_state: int,
    kmeans_n_init: int,
    minibatch_batch_size: int,
):
    """
    单次主流程与 grid 内复用：在 embedding 上拟合「主题簇」模型（KMeans 或 MiniBatchKMeans）。
    大规模消息时可用 minibatch 显著缩短聚类时间；指标可能略与全量 KMeans 不同，需在离线集上对比。
    """
    if cluster_backend == "minibatch":
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=int(minibatch_batch_size),
            n_init=3,
        )
        model.fit(X_fit)
        return model
    if cluster_backend != "kmeans":
        raise ValueError(f"Unsupported cluster_backend={cluster_backend}")
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=int(kmeans_n_init),
    )
    model.fit(X_fit)
    return model


def _normalize_fusion_weights(
    topic_weight: float,
    embed_weight: float,
    global_weight: float,
) -> Tuple[float, float, float]:
    s = abs(float(topic_weight)) + abs(float(embed_weight)) + abs(float(global_weight))
    if s < 1e-12:
        return 0.0, 0.0, 0.0
    return (
        float(topic_weight) / s,
        float(embed_weight) / s,
        float(global_weight) / s,
    )


def _best_match_score(
    sims: np.ndarray,
    agg_mode: str,
    topk: int,
    temperature: float,
) -> float:
    if sims.size == 0:
        return 0.0
    if agg_mode == "max":
        return float(np.max(sims))
    if agg_mode == "topk_mean":
        k = min(int(topk), int(sims.size))
        if k <= 0:
            return 0.0
        top_vals = np.partition(sims, -k)[-k:]
        return float(np.mean(top_vals))
    if agg_mode == "softmax":
        t = max(float(temperature), 1e-12)
        x = sims / t
        x = x - np.max(x)
        w = np.exp(x, dtype=np.float32)
        w = w / np.clip(np.sum(w), 1e-12, None)
        return float(np.sum(w * sims))
    raise ValueError(f"Unsupported agg_mode={agg_mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_need_align_accounts_csv",
        type=str,
        default=os.path.join(_DEFAULT_INPUT_DATA_DIR, "source_need_align_accounts.csv"),
    )
    parser.add_argument(
        "--target_need_align_accounts_csv",
        type=str,
        default=os.path.join(_DEFAULT_INPUT_DATA_DIR, "target_need_align_accounts.csv"),
    )
    parser.add_argument(
        "--source_alignment_ground_truth_csv",
        type=str,
        default=os.path.join(_DEFAULT_INPUT_DATA_DIR, "source_alignment_ground_truth.csv"),
    )
    parser.add_argument(
        "--target_alignment_ground_truth_csv",
        type=str,
        default=os.path.join(_DEFAULT_INPUT_DATA_DIR, "target_alignment_ground_truth.csv"),
    )
    parser.add_argument("--output_csv", type=str, default="topic_pred_top10.csv")

    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--time_window_days", type=int, default=2)
    parser.add_argument("--tz_offset_hours", type=int, default=0)
    parser.add_argument(
        "--window_direction",
        type=str,
        default="forward",
        choices=["forward", "bidirectional"],
    )

    # Topic modeling / clustering
    parser.add_argument("--num_topics", type=int, default=80)
    parser.add_argument("--cluster_random_state", type=int, default=42)
    parser.add_argument(
        "--cluster_backend",
        type=str,
        default="kmeans",
        choices=["kmeans", "minibatch"],
        help="主题簇拟合：kmeans=全量 KMeans（默认）；minibatch=MiniBatchKMeans，适合十几万级以上发言、缩短聚类时间。",
    )
    parser.add_argument(
        "--kmeans_n_init",
        type=int,
        default=10,
        help="仅 cluster_backend=kmeans 时生效，对应 KMeans 的 n_init。",
    )
    parser.add_argument(
        "--minibatch_batch_size",
        type=int,
        default=4096,
        help="仅 cluster_backend=minibatch 时生效。",
    )
    parser.add_argument("--topic_fit_scope", type=str, default="both", choices=["both", "source_only"])

    # Embedding（GPU 加速需安装带 CUDA 的 PyTorch；单卡 4070 一般用 --device cuda 或 auto）
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="embedding 设备：auto=有 CUDA 则用 GPU，否则 CPU；cpu；cuda（固定用第 0 块 GPU）；cuda:0 / cuda:1 等。",
    )
    parser.add_argument("--model_name", type=str, default="BAAI/bge-m3")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)

    # Topic assignment granularity
    parser.add_argument("--assignment_mode", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--softmax_temperature", type=float, default=0.07)
    parser.add_argument(
        "--day_match_agg",
        type=str,
        default="max",
        choices=["max", "topk_mean", "softmax"],
    )
    parser.add_argument("--day_match_topk", type=int, default=3)
    parser.add_argument("--day_match_temperature", type=float, default=0.07)

    # Multi-signal fusion weights (all scores are cosine-like, already L2 normalized where applicable)
    parser.add_argument("--topic_weight", type=float, default=0.7)
    parser.add_argument("--embed_weight", type=float, default=0.3)
    parser.add_argument("--global_weight", type=float, default=0.1)
    parser.add_argument(
        "--normalize_fusion_weights",
        action="store_true",
        help="将 topic/embed/global 三权重按 L1 归一化后再融合（不改变候选排序，仅统一分数尺度）。",
    )

    # Output formatting
    parser.add_argument("--candidate_delim", type=str, default=";")
    parser.add_argument("--run_grid_search", action="store_true")
    parser.add_argument("--grid_num_topics", type=str, default="40,80,120")
    parser.add_argument("--grid_softmax_temperature", type=str, default="0.05,0.07,0.1")
    parser.add_argument("--grid_assignment_mode", type=str, default="hard,soft")
    parser.add_argument("--grid_time_window_days", type=str, default="1,2,3")
    parser.add_argument("--grid_topic_weight", type=str, default="0.7,0.6")
    parser.add_argument("--grid_embed_weight", type=str, default="0.3,0.4")
    parser.add_argument("--grid_global_weight", type=str, default="0.1,0.0")
    parser.add_argument("--grid_output_csv", type=str, default="topic_grid_metrics.csv")

    # 子集：加速调试 / 小样本实验（不读入或不在全流程使用全部账号与发言）
    parser.add_argument(
        "--max_align_accounts",
        type=int,
        default=None,
        help="仅使用前 N 个「待对齐账号」：源/目标两侧列表均截断为前 N 条，"
        "请保证 CSV 中账号成对顺序一致（如 s_user_001 与 t_user_001 同行序）。",
    )
    parser.add_argument(
        "--max_text_rows_source",
        type=int,
        default=None,
        help="源侧 text 行数上限；超出则随机下采样（--subset_seed）。",
    )
    parser.add_argument(
        "--max_text_rows_target",
        type=int,
        default=None,
        help="目标侧 text 行数上限；超出则随机下采样。",
    )
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=42,
        help="text 行下采样随机种子。",
    )

    args = parser.parse_args()

    source_need_accounts = _load_need_align_accounts(args.source_need_align_accounts_csv)
    target_need_accounts = _load_need_align_accounts(args.target_need_align_accounts_csv)

    if args.max_align_accounts is not None and args.max_align_accounts > 0:
        n = min(args.max_align_accounts, len(source_need_accounts), len(target_need_accounts))
        source_need_accounts = source_need_accounts[:n]
        target_need_accounts = target_need_accounts[:n]
        print(
            f"[Subset] max_align_accounts={args.max_align_accounts} -> 实际使用 {n} 对账号 "
            f"（源/目标列表均取前 {n} 条）"
        )

    src_df = pd.read_csv(args.source_alignment_ground_truth_csv)
    tgt_df = pd.read_csv(args.target_alignment_ground_truth_csv)

    # Filter to need-aligned accounts.
    src_df = src_df[src_df["account_id"].astype(str).isin(source_need_accounts)]
    tgt_df = tgt_df[tgt_df["account_id"].astype(str).isin(target_need_accounts)]

    # We only use text for this module.
    src_text = src_df[src_df["type"] == "text"].copy()
    tgt_text = tgt_df[tgt_df["type"] == "text"].copy()

    if len(src_text) == 0 or len(tgt_text) == 0:
        raise ValueError("No text samples found in source/target ground truth after filtering.")

    src_text["day_id"] = _unix_day_id(src_text["time"].values, args.tz_offset_hours)
    tgt_text["day_id"] = _unix_day_id(tgt_text["time"].values, args.tz_offset_hours)

    # Ensure msg is string.
    src_text["msg"] = src_text["msg"].astype(str)
    tgt_text["msg"] = tgt_text["msg"].astype(str)

    def _maybe_downsample_text_rows(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
        if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
            return df
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=max_rows, replace=False)
        return df.iloc[sorted(idx)].reset_index(drop=True)

    if args.max_text_rows_source is not None or args.max_text_rows_target is not None:
        n0, n1 = len(src_text), len(tgt_text)
        src_text = _maybe_downsample_text_rows(src_text, args.max_text_rows_source, args.subset_seed)
        tgt_text = _maybe_downsample_text_rows(
            tgt_text, args.max_text_rows_target, args.subset_seed + 1
        )
        print(
            f"[Subset] text 行: source {n0} -> {len(src_text)}, target {n1} -> {len(tgt_text)} "
            f"(max_text_rows_source={args.max_text_rows_source}, max_text_rows_target={args.max_text_rows_target})"
        )

    # Embedding all text rows (used both for topic clustering and per-user hist).
    print(f"[Embed] source text rows: {len(src_text)}, target text rows: {len(tgt_text)}")
    src_emb = embed_texts(
        args.model_name,
        src_text["msg"].tolist(),
        args.batch_size,
        args.max_length,
        device=args.device,
    )
    tgt_emb = embed_texts(
        args.model_name,
        tgt_text["msg"].tolist(),
        args.batch_size,
        args.max_length,
        device=args.device,
    )

    src_emb = _l2_normalize(src_emb)
    tgt_emb = _l2_normalize(tgt_emb)

    # Build additional "fingerprints" from embeddings:
    # - per-user-per-day average embedding
    # - per-user global average embedding
    src_day_emb = build_user_day_embed_avg(src_text, src_emb)
    tgt_day_emb = build_user_day_embed_avg(tgt_text, tgt_emb)
    src_global_emb = build_user_global_embed_avg(src_text, src_emb)
    tgt_global_emb = build_user_global_embed_avg(tgt_text, tgt_emb)

    if args.topic_fit_scope == "source_only":
        X_fit = src_emb
    else:
        X_fit = np.vstack([src_emb, tgt_emb])

    effective_num_topics = min(int(args.num_topics), X_fit.shape[0])
    if effective_num_topics < 2:
        raise ValueError("num_topics too small compared to available text samples.")

    print(
        f"[Cluster] backend={args.cluster_backend}, num_topics={effective_num_topics}, "
        f"fit_scope={args.topic_fit_scope}, fit_embeddings={X_fit.shape[0]}"
    )
    kmeans = _fit_topic_cluster_model(
        X_fit,
        effective_num_topics,
        cluster_backend=args.cluster_backend,
        random_state=args.cluster_random_state,
        kmeans_n_init=args.kmeans_n_init,
        minibatch_batch_size=args.minibatch_batch_size,
    )

    num_topics = effective_num_topics

    print(f"[Topics] Build per-user-per-day topic distributions (text only, mode={args.assignment_mode}).")
    if args.assignment_mode == "hard":
        src_labels = kmeans.predict(src_emb)
        tgt_labels = kmeans.predict(tgt_emb)
        src_themes = build_user_day_topic_hist(src_text, src_labels, num_topics)
        tgt_themes = build_user_day_topic_hist(tgt_text, tgt_labels, num_topics)
    else:
        # Soft assignment using centroids similarity (cosine since embeddings/centers are L2-normalized).
        centers = kmeans.cluster_centers_.astype(np.float32)
        centers = _l2_normalize(centers)
        src_sims = src_emb @ centers.T
        tgt_sims = tgt_emb @ centers.T
        src_probs = _softmax(src_sims, temperature=args.softmax_temperature)
        tgt_probs = _softmax(tgt_sims, temperature=args.softmax_temperature)
        src_themes = build_user_day_topic_hist_soft(src_text, src_probs, num_topics)
        tgt_themes = build_user_day_topic_hist_soft(tgt_text, tgt_probs, num_topics)

    # Build arrays for fast scoring.
    embed_dim = src_emb.shape[1]

    def pack_account_days_multi(
        themes_by_account: Dict[str, List[Tuple[int, np.ndarray]]],
        day_emb_by_account: Dict[str, List[Tuple[int, np.ndarray]]],
        acc_id: str,
    ):
        topic_items = themes_by_account.get(acc_id, [])
        embed_items = day_emb_by_account.get(acc_id, [])

        day_to_topic = {int(d): v for d, v in topic_items}
        day_to_embed = {int(d): v for d, v in embed_items}

        all_days = sorted(set(day_to_topic.keys()) | set(day_to_embed.keys()))
        if not all_days:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, num_topics), dtype=np.float32),
                np.zeros((0, embed_dim), dtype=np.float32),
            )

        days = np.asarray(all_days, dtype=np.int64)
        topic_vecs = np.stack(
            [day_to_topic.get(int(d), np.zeros((num_topics,), dtype=np.float32)) for d in all_days],
            axis=0,
        ).astype(np.float32)
        embed_vecs = np.stack(
            [day_to_embed.get(int(d), np.zeros((embed_dim,), dtype=np.float32)) for d in all_days],
            axis=0,
        ).astype(np.float32)
        return days, topic_vecs, embed_vecs

    src_packed = {acc: pack_account_days_multi(src_themes, src_day_emb, acc) for acc in source_need_accounts}
    tgt_packed = {acc: pack_account_days_multi(tgt_themes, tgt_day_emb, acc) for acc in target_need_accounts}

    def score_source_against_targets_with_agg(
        src_days: np.ndarray,
        src_vecs: np.ndarray,
        tgt_days: np.ndarray,
        tgt_vecs: np.ndarray,
        time_window_days: int,
        window_direction: str,
        day_match_agg: str,
        day_match_topk: int,
        day_match_temperature: float,
    ) -> float:
        total = 0.0
        valid_source_day_cnt = 0
        for day_idx in range(src_vecs.shape[0]):
            day_s = src_days[day_idx]
            if window_direction == "forward":
                mask = (tgt_days >= day_s) & (tgt_days <= (day_s + time_window_days))
            else:
                mask = (tgt_days >= (day_s - time_window_days)) & (tgt_days <= (day_s + time_window_days))
            if not np.any(mask):
                continue
            sims = tgt_vecs[mask] @ src_vecs[day_idx]
            total += _best_match_score(sims, day_match_agg, day_match_topk, day_match_temperature)
            valid_source_day_cnt += 1
        if valid_source_day_cnt == 0:
            return 0.0
        return total / valid_source_day_cnt

    def run_once(
        src_packed_local: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        tgt_packed_local: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        topic_weight: float,
        embed_weight: float,
        global_weight: float,
        time_window_days: int,
    ):
        tw, ew, gw = topic_weight, embed_weight, global_weight
        if args.normalize_fusion_weights:
            tw, ew, gw = _normalize_fusion_weights(tw, ew, gw)

        predictions_rows = []
        hit1_sum = 0
        hitK_sum = 0
        mrr_sum = 0.0
        eval_cnt = 0

        for src_acc in source_need_accounts:
            src_days, src_topic_vecs, src_embed_vecs = src_packed_local[src_acc]
            scores = []
            for tgt_acc in target_need_accounts:
                tgt_days, tgt_topic_vecs, tgt_embed_vecs = tgt_packed_local[tgt_acc]
                topic_score = (
                    score_source_against_targets_with_agg(
                        src_days,
                        src_topic_vecs,
                        tgt_days,
                        tgt_topic_vecs,
                        time_window_days,
                        args.window_direction,
                        args.day_match_agg,
                        args.day_match_topk,
                        args.day_match_temperature,
                    )
                    if src_topic_vecs.shape[0] > 0 and tgt_topic_vecs.shape[0] > 0
                    else 0.0
                )
                embed_score = (
                    score_source_against_targets_with_agg(
                        src_days,
                        src_embed_vecs,
                        tgt_days,
                        tgt_embed_vecs,
                        time_window_days,
                        args.window_direction,
                        args.day_match_agg,
                        args.day_match_topk,
                        args.day_match_temperature,
                    )
                    if src_embed_vecs.shape[0] > 0 and tgt_embed_vecs.shape[0] > 0
                    else 0.0
                )
                global_score = 0.0
                if abs(gw) > 1e-15:
                    src_g = src_global_emb.get(src_acc)
                    tgt_g = tgt_global_emb.get(tgt_acc)
                    if src_g is not None and tgt_g is not None:
                        global_score = float(tgt_g @ src_g)
                score = tw * topic_score + ew * embed_score + gw * global_score
                scores.append(score)

            scores = np.asarray(scores, dtype=np.float32)
            order = np.argsort(-scores, kind="stable")
            ranked_targets = [target_need_accounts[i] for i in order]
            top_targets = ranked_targets[: args.topk]
            predictions_rows.append(
                {
                    "source_account.id": src_acc,
                    "predict.target.account.id": args.candidate_delim.join(top_targets),
                }
            )
            gt_tgt = _predict_target_id_from_source(src_acc)
            hit1, hitK, mrr = compute_metrics(ranked_targets, gt_tgt, args.topk)
            hit1_sum += hit1
            hitK_sum += hitK
            mrr_sum += mrr
            eval_cnt += 1

        hit1 = hit1_sum / max(eval_cnt, 1)
        hitK = hitK_sum / max(eval_cnt, 1)
        mrr = mrr_sum / max(eval_cnt, 1)
        return predictions_rows, hit1, hitK, mrr, eval_cnt

    def prepare_topic_views(
        num_topics_raw: int,
        assignment_mode: str,
        softmax_temperature: float,
    ):
        if args.topic_fit_scope == "source_only":
            X_fit_local = src_emb
        else:
            X_fit_local = np.vstack([src_emb, tgt_emb])
        effective_num_topics_local = min(int(num_topics_raw), X_fit_local.shape[0])
        if effective_num_topics_local < 2:
            raise ValueError("num_topics too small compared to available text samples.")

        kmeans_local = _fit_topic_cluster_model(
            X_fit_local,
            effective_num_topics_local,
            cluster_backend=args.cluster_backend,
            random_state=args.cluster_random_state,
            kmeans_n_init=args.kmeans_n_init,
            minibatch_batch_size=args.minibatch_batch_size,
        )

        if assignment_mode == "hard":
            src_labels_local = kmeans_local.predict(src_emb)
            tgt_labels_local = kmeans_local.predict(tgt_emb)
            src_themes_local = build_user_day_topic_hist(src_text, src_labels_local, effective_num_topics_local)
            tgt_themes_local = build_user_day_topic_hist(tgt_text, tgt_labels_local, effective_num_topics_local)
        else:
            centers_local = kmeans_local.cluster_centers_.astype(np.float32)
            centers_local = _l2_normalize(centers_local)
            src_sims_local = src_emb @ centers_local.T
            tgt_sims_local = tgt_emb @ centers_local.T
            src_probs_local = _softmax(src_sims_local, temperature=softmax_temperature)
            tgt_probs_local = _softmax(tgt_sims_local, temperature=softmax_temperature)
            src_themes_local = build_user_day_topic_hist_soft(src_text, src_probs_local, effective_num_topics_local)
            tgt_themes_local = build_user_day_topic_hist_soft(tgt_text, tgt_probs_local, effective_num_topics_local)

        src_packed_local = {
            acc: pack_account_days_multi(src_themes_local, src_day_emb, acc)
            for acc in source_need_accounts
        }
        tgt_packed_local = {
            acc: pack_account_days_multi(tgt_themes_local, tgt_day_emb, acc)
            for acc in target_need_accounts
        }
        return src_packed_local, tgt_packed_local, effective_num_topics_local

    if not args.run_grid_search:
        print("[Run] 单次主流程：1×embed → 1×cluster → 1×score（未使用 --run_grid_search）")
        if args.normalize_fusion_weights:
            _tw, _ew, _gw = _normalize_fusion_weights(
                args.topic_weight, args.embed_weight, args.global_weight
            )
            print(
                f"[Fusion] L1 归一化后权重: topic={_tw:.4f}, embed={_ew:.4f}, global={_gw:.4f}"
            )
        print("[Predict] Scoring source accounts vs all target accounts...")
        predictions_rows, hit1, hitK, mrr, eval_cnt = run_once(
            src_packed_local=src_packed,
            tgt_packed_local=tgt_packed,
            topic_weight=args.topic_weight,
            embed_weight=args.embed_weight,
            global_weight=args.global_weight,
            time_window_days=args.time_window_days,
        )
        pred_df = pd.DataFrame(predictions_rows)
        pred_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"[Output] saved predictions to: {args.output_csv}")
        print(f"[Metrics] eval_cnt={eval_cnt} Hit@1={hit1:.6f} Hit@{args.topk}={hitK:.6f} MRR={mrr:.6f}")
    else:
        grid_num_topics = _parse_csv_ints(args.grid_num_topics)
        grid_assignment_modes = [x.strip() for x in str(args.grid_assignment_mode).split(",") if x.strip()]
        grid_softmax_temps = _parse_csv_floats(args.grid_softmax_temperature)
        grid_windows = _parse_csv_ints(args.grid_time_window_days)
        grid_topic_ws = _parse_csv_floats(args.grid_topic_weight)
        grid_embed_ws = _parse_csv_floats(args.grid_embed_weight)
        grid_global_ws = _parse_csv_floats(args.grid_global_weight)

        rows = []
        print("[Grid] Run reproducible ablation/grid over params...")
        for n_topics in grid_num_topics:
            for assign_mode in grid_assignment_modes:
                for soft_t in grid_softmax_temps:
                    if assign_mode == "hard":
                        soft_t = args.softmax_temperature
                    src_packed_local, tgt_packed_local, eff_topics = prepare_topic_views(
                        num_topics_raw=n_topics,
                        assignment_mode=assign_mode,
                        softmax_temperature=soft_t,
                    )
                    for tw in grid_windows:
                        for tw_topic in grid_topic_ws:
                            for tw_embed in grid_embed_ws:
                                for tw_global in grid_global_ws:
                                    if abs(tw_topic) + abs(tw_embed) + abs(tw_global) < 1e-12:
                                        continue
                                    predictions_rows, hit1, hitK, mrr, eval_cnt = run_once(
                                        src_packed_local=src_packed_local,
                                        tgt_packed_local=tgt_packed_local,
                                        topic_weight=tw_topic,
                                        embed_weight=tw_embed,
                                        global_weight=tw_global,
                                        time_window_days=tw,
                                    )
                                    _ = predictions_rows
                                    rows.append(
                                        {
                                            "num_topics": eff_topics,
                                            "assignment_mode": assign_mode,
                                            "softmax_temperature": soft_t,
                                            "time_window_days": tw,
                                            "topic_weight": tw_topic,
                                            "embed_weight": tw_embed,
                                            "global_weight": tw_global,
                                            "window_direction": args.window_direction,
                                            "day_match_agg": args.day_match_agg,
                                            "day_match_topk": args.day_match_topk,
                                            "day_match_temperature": args.day_match_temperature,
                                            "eval_cnt": eval_cnt,
                                            "hit1": hit1,
                                            "hitK": hitK,
                                            "mrr": mrr,
                                        }
                                    )
        grid_df = pd.DataFrame(rows)
        if len(grid_df) > 0:
            grid_df = grid_df.sort_values(["hit1", "hitK", "mrr"], ascending=[False, False, False]).reset_index(drop=True)
        grid_df.to_csv(args.grid_output_csv, index=False, encoding="utf-8-sig")
        print(f"[Grid] saved to: {args.grid_output_csv}, rows={len(grid_df)}")


if __name__ == "__main__":
    main()

