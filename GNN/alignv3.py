"""
用户身份对齐系统 v2 (重构版)
==============================
核心函数 run_alignment() 接收所有参数, 返回 (输出文件路径, 评估指标字典)。

输出 CSV 两列:
  - source_account_id
  - predict_target_account_id  (逗号分隔的 top-K 候选, 排名靠前在前)

评估: hit@1, hit@K, MRR

Python 3.10 | 无需训练神经网络
"""

import os
import sys
import time
import datetime
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


# ============================================================
# 二部图
# ============================================================
class BipartiteGraph:

    def __init__(self, df: pd.DataFrame):
        self.user2groups: dict[str, set] = defaultdict(set)
        self.group2users: dict[str, set] = defaultdict(set)

        edges = df[["account_id", "room_id"]].drop_duplicates()
        for _, row in edges.iterrows():
            u, g = row["account_id"], row["room_id"]
            self.user2groups[u].add(g)
            self.group2users[g].add(u)

        self.edge_weight: dict = (
            df.groupby(["account_id", "room_id"]).size().to_dict()
        )
        self.users = list(self.user2groups.keys())
        self.groups = list(self.group2users.keys())
        print(
            f"    Graph: {len(self.users)} users, "
            f"{len(self.groups)} groups, "
            f"{len(self.edge_weight)} edges"
        )

    def get_user_group_weight(self, user, group) -> int:
        return self.edge_weight.get((user, group), 0)

    def get_group_total_msgs(self, group) -> int:
        return sum(
            self.edge_weight.get((u, group), 0)
            for u in self.group2users[group]
        )


# ============================================================
# GNN 消息传递
# ============================================================
class GNNAggregator:

    def __init__(self, graph: BipartiteGraph, n_layers: int = 3):
        self.graph = graph
        self.n_layers = n_layers

        self.group_agg_w: dict = {}
        for g in graph.groups:
            total = graph.get_group_total_msgs(g)
            if total > 0:
                for u in graph.group2users[g]:
                    self.group_agg_w[(u, g)] = (
                        graph.get_user_group_weight(u, g) / total
                    )

        self.user_agg_w: dict = {}
        for u in graph.users:
            total = sum(
                graph.get_user_group_weight(u, g)
                for g in graph.user2groups[u]
            )
            if total > 0:
                for g in graph.user2groups[u]:
                    self.user_agg_w[(u, g)] = (
                        graph.get_user_group_weight(u, g) / total
                    )

    def propagate(self, user_features: dict, feat_dim: int) -> dict:
        h = {
            u: user_features.get(u, np.zeros(feat_dim)).copy()
            for u in self.graph.users
        }
        zero = np.zeros(feat_dim)

        for _ in range(self.n_layers):
            h_group = {}
            for g in self.graph.groups:
                agg = np.zeros(feat_dim)
                for u in self.graph.group2users[g]:
                    agg += self.group_agg_w.get((u, g), 0) * h.get(u, zero)
                h_group[g] = agg

            alpha = 0.6
            h_new = {}
            for u in self.graph.users:
                groups = self.graph.user2groups[u]
                if not groups:
                    h_new[u] = h[u]
                    continue
                nb = np.zeros(feat_dim)
                for g in groups:
                    nb += self.user_agg_w.get((u, g), 0) * h_group[g]
                h_new[u] = alpha * h[u] + (1 - alpha) * nb
            h = h_new

        for u in h:
            n = np.linalg.norm(h[u])
            if n > 0:
                h[u] /= n
        return h


# ============================================================
# 特征提取器
# ============================================================
class FeatureExtractor:

    def __init__(self, df: pd.DataFrame, label: str = ""):
        self.df = df
        self.label = label
        self.users = sorted(df["account_id"].unique())
        self.user_idx = {u: i for i, u in enumerate(self.users)}

    def text_tfidf(self, vectorizer=None):
        user_docs = (
            self.df.groupby("account_id")["msg"]
            .apply(lambda x: " ".join(x.astype(str)))
        )
        docs = [user_docs.get(u, "") for u in self.users]

        if vectorizer is None:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=8000,
                sublinear_tf=True,
                min_df=2,
                max_df=0.95,
            )
            tfidf = vectorizer.fit_transform(docs)
        else:
            tfidf = vectorizer.transform(docs)

        print(f"    [{self.label}] TF-IDF shape: {tfidf.shape}")
        return tfidf, vectorizer

    def text_svd(self, tfidf, n_components: int = 64):
        n_comp = min(n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        dense = svd.fit_transform(tfidf)
        return {u: dense[i] for i, u in enumerate(self.users)}, n_comp

    def temporal_features(self, hour_bins: int = 24, dow_bins: int = 7):
        feats = {}
        for uid, grp in self.df.groupby("account_id"):
            h_hist = np.zeros(hour_bins)
            d_hist = np.zeros(dow_bins)
            for ts in grp["time"].dropna():
                try:
                    dt = datetime.datetime.fromtimestamp(int(ts))
                    h_hist[dt.hour] += 1
                    d_hist[dt.weekday()] += 1
                except Exception:
                    continue
            hs, ds = h_hist.sum(), d_hist.sum()
            if hs > 0:
                h_hist /= hs
            if ds > 0:
                d_hist /= ds
            feats[uid] = np.concatenate([h_hist, d_hist])
        return feats

    def group_fingerprint(self):
        group_sizes = self.df.groupby("room_id")["account_id"].nunique().to_dict()
        ug_cnt = self.df.groupby(["account_id", "room_id"]).size().to_dict()

        feats = {}
        for uid, grp in self.df.groupby("account_id"):
            groups = grp["room_id"].unique()
            sizes = [group_sizes.get(g, 0) for g in groups]
            ratios = []
            for g in groups:
                total = sum(
                    ug_cnt.get((u2, g), 0)
                    for u2 in self.df[self.df["room_id"] == g]["account_id"].unique()
                )
                ratios.append(ug_cnt.get((uid, g), 0) / max(total, 1))

            top5 = sorted(sizes, reverse=True)[:5]
            top5 += [0] * (5 - len(top5))

            feats[uid] = np.array(
                [
                    np.log1p(len(groups)),
                    np.mean(sizes) if sizes else 0,
                    np.std(sizes) if len(sizes) > 1 else 0,
                    max(sizes) if sizes else 0,
                    np.mean(ratios) if ratios else 0,
                    np.std(ratios) if len(ratios) > 1 else 0,
                ]
                + top5
            )
        return feats

    def activity_features(self):
        feats = {}
        for uid, grp in self.df.groupby("account_id"):
            n_msgs = len(grp)
            lens = grp["msg"].astype(str).str.len()
            ts = grp["time"]
            span = ts.max() - ts.min()
            freq = n_msgs / (span / 86400) if span > 0 else float(n_msgs)
            std_val = lens.std()
            feats[uid] = np.array(
                [
                    np.log1p(n_msgs),
                    np.log1p(grp["room_id"].nunique()),
                    np.log1p(lens.mean()),
                    np.log1p(std_val if not np.isnan(std_val) else 0),
                    np.log1p(span),
                    np.log1p(freq),
                ]
            )
        return feats


# ============================================================
# 内部工具函数
# ============================================================
def _read_csv(filepath: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return pd.read_csv(filepath, encoding=enc, on_bad_lines="skip")
        except Exception:
            continue
    raise RuntimeError(f"Cannot read {filepath}")


def _load_text_data(filepath: str) -> pd.DataFrame:
    print(f"  Loading: {filepath}")
    df = _read_csv(filepath)
    df = df[df["type"] == "text"].reset_index(drop=True)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    print(
        f"    -> {len(df)} text msgs, "
        f"{df['account_id'].nunique()} users, "
        f"{df['room_id'].nunique()} groups"
    )
    return df


def _load_accounts(filepath: str) -> list:
    df = _read_csv(filepath)
    col = "account_id" if "account_id" in df.columns else df.columns[0]
    accounts = df[col].unique().tolist()
    print(f"  Need-align accounts: {len(accounts)}")
    return accounts


# ============================================================
# 对齐计算 (内部)
# ============================================================
def _compute_alignment(
    src_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    need_align_accounts: list,
    K: int,
    gnn_layers: int,
    svd_dim: int,
    weights: dict,
) -> pd.DataFrame:

    hour_bins, dow_bins = 24, 7

    # 1. 二部图
    print("\n" + "=" * 60)
    print("STEP 1: Build Bipartite Graphs")
    print("=" * 60)
    print("  [Source]")
    src_graph = BipartiteGraph(src_df)
    print("  [Target]")
    tgt_graph = BipartiteGraph(tgt_df)

    # 2. 特征
    print("\n" + "=" * 60)
    print("STEP 2: Extract Features")
    print("=" * 60)
    src_ext = FeatureExtractor(src_df, "SRC")
    tgt_ext = FeatureExtractor(tgt_df, "TGT")

    src_tfidf, vectorizer = src_ext.text_tfidf()
    tgt_tfidf, _ = tgt_ext.text_tfidf(vectorizer=vectorizer)

    src_text_dense, feat_dim = src_ext.text_svd(src_tfidf, svd_dim)
    tgt_text_dense, _ = tgt_ext.text_svd(tgt_tfidf, svd_dim)

    src_temporal = src_ext.temporal_features(hour_bins, dow_bins)
    tgt_temporal = tgt_ext.temporal_features(hour_bins, dow_bins)
    src_gfp = src_ext.group_fingerprint()
    tgt_gfp = tgt_ext.group_fingerprint()
    src_act = src_ext.activity_features()
    tgt_act = tgt_ext.activity_features()

    # 3. GNN
    print("\n" + "=" * 60)
    print(f"STEP 3: GNN ({gnn_layers}-layer message passing)")
    print("=" * 60)
    print("  Source propagation...")
    src_gnn_feats = GNNAggregator(src_graph, gnn_layers).propagate(
        src_text_dense, feat_dim
    )
    print("  Target propagation...")
    tgt_gnn_feats = GNNAggregator(tgt_graph, gnn_layers).propagate(
        tgt_text_dense, feat_dim
    )

    # 4. 融合 & 排序
    print("\n" + "=" * 60)
    print("STEP 4: Multi-Signal Fusion & Ranking")
    print("=" * 60)

    tgt_users = tgt_ext.users
    n_tgt = len(tgt_users)
    temporal_dim = hour_bins + dow_bins
    gfp_dim, act_dim = 11, 6

    def _mat(feat_dict, users, dim):
        mat = np.zeros((len(users), dim))
        for i, u in enumerate(users):
            if u in feat_dict:
                v = feat_dict[u]
                d = min(len(v), dim)
                mat[i, :d] = v[:d]
        return mat

    tgt_gnn_mat = normalize(_mat(tgt_gnn_feats, tgt_users, feat_dim))
    tgt_tmp_mat = normalize(_mat(tgt_temporal, tgt_users, temporal_dim))
    tgt_gfp_mat = normalize(_mat(tgt_gfp, tgt_users, gfp_dim))
    tgt_act_mat = normalize(_mat(tgt_act, tgt_users, act_dim))

    src_users = src_ext.users
    src_uid_idx = {u: i for i, u in enumerate(src_users)}
    valid = [u for u in need_align_accounts if u in src_uid_idx]
    print(f"  Aligning {len(valid)} source → {n_tgt} target users  (K={K})")

    def _add_signal(feat_dict, tgt_mat, uid, key):
        if uid not in feat_dict:
            return 0.0
        v = feat_dict[uid]
        n = np.linalg.norm(v)
        return weights[key] * (tgt_mat @ (v / n)) if n > 0 else 0.0

    rows = []
    for s_uid in valid:
        scores = np.zeros(n_tgt)

        s_idx = src_uid_idx[s_uid]
        scores += weights["text_self"] * cosine_similarity(
            src_tfidf[s_idx], tgt_tfidf
        ).flatten()

        scores += _add_signal(src_gnn_feats, tgt_gnn_mat, s_uid, "text_gnn")
        scores += _add_signal(src_temporal, tgt_tmp_mat, s_uid, "temporal")
        scores += _add_signal(src_gfp, tgt_gfp_mat, s_uid, "group_struct")
        scores += _add_signal(src_act, tgt_act_mat, s_uid, "activity")

        top_idx = np.argsort(scores)[-K:][::-1]
        candidates = [str(tgt_users[i]) for i in top_idx]
        rows.append(
            {
                "source_account_id": s_uid,
                "predict_target_account_id": ",".join(candidates),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 评估 (内部)
# ============================================================
def _evaluate(result_df: pd.DataFrame, gt_file: str, K: int) -> dict:
    if not os.path.exists(gt_file):
        print("\n  [Ground truth file not found — skipping evaluation]")
        return {}

    gt = _read_csv(gt_file)
    src_col = "source_account" if "source_account" in gt.columns else gt.columns[0]
    tgt_col = "target_account" if "target_account" in gt.columns else gt.columns[1]
    gt_map = dict(zip(gt[src_col], gt[tgt_col].astype(str)))
    print(f"\n  Ground truth pairs: {len(gt_map)}")

    hit1, hitK, mrr_sum, n_eval = 0, 0, 0.0, 0

    for _, row in result_df.iterrows():
        s_uid = row["source_account_id"]
        if s_uid not in gt_map:
            continue
        expected = gt_map[s_uid]
        candidates = str(row["predict_target_account_id"]).split(",")
        n_eval += 1
        if expected in candidates:
            rank = candidates.index(expected) + 1
            mrr_sum += 1.0 / rank
            if rank == 1:
                hit1 += 1
            if rank <= K:
                hitK += 1

    if n_eval == 0:
        print("  No overlapping users between results and ground truth.")
        return {}

    metrics = {
        "hit@1": hit1 / n_eval,
        f"hit@{K}": hitK / n_eval,
        "MRR": mrr_sum / n_eval,
        "evaluated_users": n_eval,
    }

    print(f"\n  Evaluated on {n_eval} users")
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-' * 32}")
    for name, val in metrics.items():
        if isinstance(val, float):
            print(f"  {name:<20} {val:>10.4f}")
        else:
            print(f"  {name:<20} {val:>10}")

    return metrics


# ============================================================
# ★ 核心对外函数
# ============================================================
def run_alignment(
    source_file: str,
    target_file: str,
    need_align_file: str,
    output_file: str,
    K: int = 10,
    ground_truth_file: Optional[str] = None,
    gnn_layers: int = 3,
    svd_dim: int = 64,
    weights: Optional[dict] = None,
) -> tuple[str, dict]:
    """
    用户身份对齐核心入口。

    Parameters
    ----------
    source_file       : 源平台消息 CSV 路径
    target_file       : 目标平台消息 CSV 路径
    need_align_file   : 待对齐账户列表 CSV 路径
    output_file       : 结果输出 CSV 路径
    K                 : top-K 候选数量, 同时也是 hit@K 的 K
    ground_truth_file : ground truth CSV 路径 (可选, 为 None 则跳过评估)
    gnn_layers        : GNN 消息传递层数
    svd_dim           : SVD 降维维度
    weights           : 多信号融合权重 dict, 默认:
                        {"text_self":0.25, "text_gnn":0.30, "temporal":0.15,
                         "group_struct":0.20, "activity":0.10}

    Returns
    -------
    (output_file_path: str, metrics: dict)
        output_file_path : 写出的 CSV 路径
        metrics          : {"hit@1": float, "hit@{K}": float, "MRR": float,
                            "evaluated_users": int}
                           若无 ground truth 则为空 dict {}
    """
    if weights is None:
        weights = {
            "text_self": 0.25,
            "text_gnn": 0.30,
            "temporal": 0.15,
            "group_struct": 0.20,
            "activity": 0.10,
        }

    t0 = time.time()

    print("=" * 60)
    print(f"  User Identity Alignment v2  (K={K})")
    print("  GNN Two-Phase Message Passing")
    print("=" * 60)

    for f in (source_file, target_file, need_align_file):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")

    # 加载数据
    print("\n--- Loading Data ---")
    src_df = _load_text_data(source_file)
    tgt_df = _load_text_data(target_file)
    need_align = _load_accounts(need_align_file)

    # 计算对齐
    result_df = _compute_alignment(
        src_df, tgt_df, need_align,
        K=K,
        gnn_layers=gnn_layers,
        svd_dim=svd_dim,
        weights=weights,
    )

    # 保存结果
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n--- Results saved: {output_file}  ({len(result_df)} rows) ---")

    # 评估
    metrics: dict = {}
    if ground_truth_file:
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        metrics = _evaluate(result_df, ground_truth_file, K)

    elapsed = time.time() - t0
    print(f"\n--- Total time: {elapsed:.1f}s ---")

    # 预览
    print("\n--- Preview (first 5 rows) ---")
    print(result_df.head(5).to_string(index=False))

    return output_file, metrics


# ============================================================
# CLI 入口 (直接运行时使用默认路径, 方便调试)
# ============================================================
if __name__ == "__main__":
    output_path, eval_metrics = run_alignment(
        source_file="source_alignment_ground_truth.csv",
        target_file="target_alignment_ground_truth.csv",
        need_align_file="source_need_align_accounts.csv",
        output_file="alignment_results.csv",
        K=10,
        ground_truth_file="ground_truth_mapping.csv",
    )
    print(f"\n>>> Output file: {output_path}")
    print(f">>> Metrics: {eval_metrics}")