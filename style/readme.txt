1.环境配置
pip install pandas numpy scikit-learn

2.模块
align_common.py   数据读取、相似度矩阵、Top-K、指标评估、CSV 写出
align_frequency.py 构建时间分布特征并运行对齐
align_frequency.py  构建文本/行为风格特征并运行对齐
user_alignment.py  解析参数，调用双维度对齐，输出结果与评估指标

3.维度
（1）频率维度
24h发文时段直方图  24 维
粗粒度时段分布：凌晨（0-5）/上午（6-11）/下午（12-17）/晚间（18-23）   4 维
发文时刻均值 & 标准差  2 维
（2）风格
消息长度 均值、标准差、长 （>50）/ 短消息（≤10）占比 4维
标点 标点比例/句末标点/低标点标识（5%做划分线） 3维
语气词 笑声词/感叹 2维
TF-IDF 文本向量

4.调用与传参
（1）终端：
python user_alignment.py \
  --output_dir /root/autodl-tmp/align_new_wx/time_style \
  --gt_csv /root/autodl-tmp/align_new_wx/data/align_accounts.csv \
  --k 10 \
  --max_tfidf_features 2000

（加入下面两个指令   会混合frequency和time的结果输出）
  --use_fusion
 --fusion_save_csv 

说明：
  (1)--k 为 Top-K（默认 10）
  (2) --gt_csv为 ground truth 表，用于计算 hit@1、hit@K、MRR；不传则跳过指标
  (3)--max_tfidf_features`默认 2000（同时 style_tfidf_weight=3.0、style_basic_weight=0.3、tfidf_agg=mean、tfidf_ngram_min=3、tfidf_ngram_max=5、tfidf_min_df=2 已作为默认调优）
  (4) 如不想使用 TF-IDF,仅用非文本风格特征，加 --style_no_tfidf

输出位置：
  --output_dir 下会生成：
    time_result/alignment_topk_k{K}.csv
    style_result/alignment_topk_k{K}.csv


（2）Python调用：
from align_frequency import run_frequency_alignment
from align_style import run_style_alignment
freq_cand = run_frequency_alignment(df_src, df_tgt, k=10, w24=1.0, w_coarse=1.0, w_peak=1.0)
style_cand = run_style_alignment(
    df_src,
    df_tgt,
    k=10,
    max_tfidf_features=2000,
    use_tfidf=True,
    basic_weight=0.3,
    tfidf_weight=3.0,
    tfidf_agg="mean",
    tfidf_ngram_min=3,
    tfidf_ngram_max=5,
    tfidf_min_df=2,
)



(1) 频率维度：`run_frequency_alignment(df_src, df_tgt, k, w24, w_coarse, w_peak, ...)`
`k`：每个 `source_account` 保留的 topK 候选数量。
`w24`：24h 小时分布（24维直方图）在相似度中的权重。
`w_coarse`：粗粒度时间段分布（4维：夜间/上午/下午/晚间）权重。
`w_peak`：峰值时间相关特征权重。

 可选参数：
  `smooth_sigma`：24h 分布高斯平滑。
  `hist_alpha`：分布锐化/平滑。
  `night_end/morning_end/afternoon_end`：粗粒度分段边界。
  `w_dow`：day-of-week（星期）7维直方图权重。
  `tz_offset_hours`：unix 时间戳转成本地时间的时区偏移。

(2) 风格维度：`run_style_alignment(df_src, df_tgt, k, max_tfidf_features, use_tfidf, basic_weight, tfidf_weight,tfidf_agg, tfidf_ngram_min,tfidf_ngram_max,tfidf_min_df,...)`
`k`：每个 source 保留的 topK 数量。
`max_tfidf_features`：TF-IDF 特征维度上限。
`use_tfidf`：是否启用文本 TF-IDF 特征块。
`basic_weight`：非文本风格特征块权重（长度、标点、media 比例、语气词等）。
`tfidf_weight`：文本 TF-IDF 特征块权重。
`tfidf_agg`：用户多条消息的 TF-IDF 聚合方式（`mean` 或 `concat`）。
`tfidf_ngram_min/tfidf_ngram_max`：字符 n-gram 范围。
`tfidf_min_df`：n-gram 最小出现次数/最小占比阈值。
可选参数：
  `long_len_thresh/short_len_thresh/low_punct_ratio_thresh`：长度与标点阈值。
  `tfidf_sublinear_tf`：sublinear TF 开关（缓解长文本影响）。
  `tfidf_max_msgs`：每用户最多使用多少条消息计算 TF-IDF（提速）。



df_src  ——— 源平台
df_tgt  ——— 目标平台
