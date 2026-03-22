"""
测试数据生成器
===============
生成4个文件用于测试用户对齐效果:
- source_alignment_ground_truth.csv
- target_alignment_ground_truth.csv
- source_need_align_accounts.csv
- ground_truth_mapping.csv (评估用)
"""
import pandas as pd
import numpy as np
import random
import datetime

random.seed(42)
np.random.seed(42)

# 参数
N_ALIGNED = 50           # 同一人有两个平台账号
N_SRC_ONLY = 50          # 仅在source的用户
N_TGT_ONLY = 50          # 仅在target的用户
N_SRC_GROUPS = 20
N_TGT_GROUPS = 20
N_ALIGNED_GROUPS = 12    # 两平台对应的群数
MSGS_PER_USER = (15, 50) # 每人消息数范围
TIME_START = 1735689600  # 2025-01-01
TIME_END = 1772000000    # 2026-02

# ========================================
# 消息模板: 同一人在source/target用不同措辞说同一类话
# ========================================
TOPICS = {
    "food_cn": {
        "src": ["今天吃了火锅太爽了", "这家牛肉面真不错", "周末约烧烤吗",
                "最近迷上做甜点了", "奶茶续命中", "减肥失败又吃了炸鸡",
                "新开的奶茶店味道不错", "早餐吃了豆浆油条好满足"],
        "tgt": ["火锅真的太好吃了吧", "发现一家超好吃的面馆", "这周末一起撸串",
                "在家做蛋糕翻车了哈哈", "没有奶茶活不下去", "说好的减肥又点了外卖",
                "那家新奶茶店去过了还行", "豆浆油条是早餐的灵魂"],
    },
    "work_en": {
        "src": ["Monday meeting, feeling the pressure.", "This project is finally wrapping up.",
                "Worked overtime till 10, exhausted.", "Client meeting went well today.",
                "The new colleague seems nice.", "Business trip next week, so tired.",
                "KPI deadline is killing me.", "Finally got promoted!"],
        "tgt": ["Weekly meeting was intense today.", "Project deadline almost here, wrapping up.",
                "Another overtime night, can't take it.", "Client was in good mood today.",
                "New teammate is pretty cool.", "Gotta travel again for work.",
                "These quarterly targets are insane.", "Got the promotion, so happy!"],
    },
    "life_cn": {
        "src": ["阳光明媚心情也跟着变好了", "周末去爬山了累并快乐着", "失眠了数羊也没用",
                "下雨天适合宅在家看剧", "养了只猫每天被治愈", "今天跑了五公里打卡",
                "搬家好累但新房子真不错", "好久没见老朋友了约一下"],
        "tgt": ["今天天气真好适合出去走走", "爬山虽然累但是值得", "又失眠了好烦",
                "雨天窝在家追剧最舒服", "我家猫今天又拆家了", "坚持跑步第30天",
                "终于搬完家了太折腾了", "好想念大学同学们"],
    },
    "entertainment_mix": {
        "src": ["这首歌太好听了单曲循环中", "昨晚的剧太好看了一口气追完",
                "有人一起打游戏吗", "最近在看一本推理小说",
                "演唱会的票抢到了", "新出的手游还不错"],
        "tgt": ["This song is on repeat so good", "追剧追到停不下来",
                "谁来组队打游戏", "推理小说真的很上头",
                "终于抢到演唱会门票了", "下了个新游戏还挺好玩"],
    },
    "festival": {
        "src": ["元宵节快乐吃汤圆了吗", "中秋快乐月饼吃了吗", "新年快乐万事如意",
                "端午节安康粽子走起", "国庆七天打算去旅游"],
        "tgt": ["Happy Lantern Festival!", "中秋节快乐今晚赏月",
                "Happy New Year新年好", "端午快乐你们吃甜粽还是咸粽",
                "National holiday planning done"],
    },
    "tech": {
        "src": ["Python真的太好用了", "这个bug找了一天终于修好了", "新买的MacBook体验不错",
                "AI发展太快了跟不上了", "学了一下Docker还挺有意思", "数据库又崩了头大"],
        "tgt": ["Python是最好的语言", "debug了一整天终于搞定了", "MacBook Pro真的生产力工具",
                "The pace of AI is insane", "Docker学起来还行", "数据库挂了在线等急"],
    },
    "random_cn": {
        "src": ["哈哈哈笑死我了", "收到收到", "好的好的", "666", "太卷了",
                "摸鱼中", "冲冲冲", "明天见"],
        "tgt": ["笑死了哈哈", "OK收到", "没问题", "厉害了", "卷不动了",
                "划水中", "加油加油", "See you tomorrow"],
    },
}

NOISE_SRC = ["今天股票又跌了", "驾照终于考过了", "健身房续了年卡",
             "快递到了但没人在家", "空调坏了热死了", "学会了做红烧肉",
             "同学聚会好开心", "手机屏幕摔碎了", "飞机延误三小时", "堵车堵到怀疑人生"]
NOISE_TGT = ["房价什么时候降啊", "终于拿到驾照了", "今天去跑步了",
             "快递怎么还没到", "热到融化", "试着做了顿大餐",
             "老同学们都变了好多", "手机又该换了", "航班delay了", "路上太堵了"]


def gen_timestamp(active_hour, active_days):
    """生成符合用户习惯的时间戳"""
    base = random.randint(TIME_START, TIME_END)
    dt = datetime.datetime.fromtimestamp(base)
    hour = int(np.clip(random.gauss(active_hour, 3), 0, 23))
    dt = dt.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))
    if random.random() < 0.75 and active_days:
        target_dow = random.choice(active_days)
        dt += datetime.timedelta(days=(target_dow - dt.weekday()) % 7)
    return int(dt.timestamp())


def generate():
    print("Generating test data...")
    topic_names = list(TOPICS.keys())

    # 分配用户ID (随机编号, 不让ID暴露对应关系)
    src_ids = [f"s_user_{i:03d}" for i in random.sample(range(1, 800), N_ALIGNED + N_SRC_ONLY)]
    tgt_ids = [f"t_user_{i:03d}" for i in random.sample(range(1, 800), N_ALIGNED + N_TGT_ONLY)]

    # 前N_ALIGNED个是对齐对: src_ids[i] <-> tgt_ids[i]
    aligned_src = src_ids[:N_ALIGNED]
    aligned_tgt = tgt_ids[:N_ALIGNED]
    only_src = src_ids[N_ALIGNED:]
    only_tgt = tgt_ids[N_ALIGNED:]

    # 群组 (对齐群: src_group_i <-> tgt_group_i)
    src_groups = [f"src_grp_{i:03d}" for i in range(N_SRC_GROUPS)]
    tgt_groups = [f"tgt_grp_{i:03d}" for i in range(N_TGT_GROUPS)]
    # 对齐群的映射
    aligned_sg = src_groups[:N_ALIGNED_GROUPS]
    aligned_tg = tgt_groups[:N_ALIGNED_GROUPS]

    # 给每个source用户分配profile
    profiles = {}
    for uid in src_ids:
        n_topics = random.randint(2, 4)
        fav_topics = random.sample(topic_names, n_topics)
        active_hour = random.gauss(14, 4) % 24
        active_days = sorted(random.sample(range(7), random.randint(3, 6)))
        n_grp = random.randint(2, 5)
        grps = random.sample(src_groups, n_grp)
        profiles[uid] = {
            "topics": fav_topics, "hour": active_hour,
            "days": active_days, "groups": grps,
        }

    # 对齐用户: target继承source的profile (加噪声)
    for s_uid, t_uid in zip(aligned_src, aligned_tgt):
        sp = profiles[s_uid]
        # target群: 把source的对齐群映射过去 + 一点随机
        t_grps = []
        for sg in sp["groups"]:
            idx = src_groups.index(sg) if sg in aligned_sg else -1
            if idx >= 0 and idx < len(aligned_tg):
                t_grps.append(aligned_tg[idx])
            elif random.random() < 0.3:
                t_grps.append(random.choice(tgt_groups))
        if not t_grps:
            t_grps.append(random.choice(tgt_groups))
        # 额外加1个随机群 (噪声)
        if random.random() < 0.4:
            t_grps.append(random.choice(tgt_groups))
        t_grps = list(set(t_grps))

        profiles[t_uid] = {
            "topics": sp["topics"],                              # 同样的主题偏好
            "hour": (sp["hour"] + random.gauss(0, 1.5)) % 24,   # 活跃时间微偏
            "days": sp["days"],                                  # 同样的星期偏好
            "groups": t_grps,
        }

    # 非对齐的target用户: 独立profile
    for uid in only_tgt:
        n_topics = random.randint(2, 4)
        fav_topics = random.sample(topic_names, n_topics)
        active_hour = random.gauss(14, 4) % 24
        active_days = sorted(random.sample(range(7), random.randint(3, 6)))
        n_grp = random.randint(2, 5)
        grps = random.sample(tgt_groups, n_grp)
        profiles[uid] = {
            "topics": fav_topics, "hour": active_hour,
            "days": active_days, "groups": grps,
        }

    # 生成消息
    def gen_msgs(uid, platform):
        p = profiles[uid]
        records = []
        n = random.randint(*MSGS_PER_USER)
        side = "src" if platform == "source" else "tgt"
        noise = NOISE_SRC if platform == "source" else NOISE_TGT

        for _ in range(n):
            grp = random.choice(p["groups"])
            topic = random.choice(p["topics"])
            pool = TOPICS[topic][side]
            msg = random.choice(pool) if random.random() < 0.8 else random.choice(noise)
            # 偶尔加后缀
            if random.random() < 0.12:
                msg += random.choice(["哈哈", "!", "～", " haha", " lol", "😂", "👍"])
            ts = gen_timestamp(p["hour"], p["days"])
            records.append({"account_id": uid, "room_id": grp,
                            "msg": msg, "time": ts, "type": "text"})

        # 一些非text消息
        for _ in range(random.randint(3, 10)):
            grp = random.choice(p["groups"])
            ts = gen_timestamp(p["hour"], p["days"])
            records.append({"account_id": uid, "room_id": grp,
                            "msg": random.randbytes(16).hex(),
                            "time": ts,
                            "type": random.choice(["photo", "video"])})
        return records

    src_records, tgt_records = [], []
    for uid in src_ids:
        src_records.extend(gen_msgs(uid, "source"))
    for uid in aligned_tgt + only_tgt:
        tgt_records.extend(gen_msgs(uid, "target"))

    random.shuffle(src_records)
    random.shuffle(tgt_records)

    src_df = pd.DataFrame(src_records)
    tgt_df = pd.DataFrame(tgt_records)

    src_df.to_csv("source_alignment_ground_truth.csv", index=False, encoding="utf-8-sig")
    tgt_df.to_csv("target_alignment_ground_truth.csv", index=False, encoding="utf-8-sig")

    # need_align: 所有aligned source + 20个non-aligned source
    need_align = aligned_src + random.sample(only_src, min(20, len(only_src)))
    random.shuffle(need_align)
    pd.DataFrame({"account_id": need_align}).to_csv(
        "source_need_align_accounts.csv", index=False, encoding="utf-8-sig")

    # ground truth
    gt = pd.DataFrame({"source_account": aligned_src, "target_account": aligned_tgt})
    gt.to_csv("ground_truth_mapping.csv", index=False, encoding="utf-8-sig")

    n_src_text = src_df[src_df["type"] == "text"].shape[0]
    n_tgt_text = tgt_df[tgt_df["type"] == "text"].shape[0]
    print(f"  source: {len(src_df)} records ({n_src_text} text), "
          f"{src_df['account_id'].nunique()} users, {src_df['room_id'].nunique()} groups")
    print(f"  target: {len(tgt_df)} records ({n_tgt_text} text), "
          f"{tgt_df['account_id'].nunique()} users, {tgt_df['room_id'].nunique()} groups")
    print(f"  need_align: {len(need_align)}, ground_truth: {len(gt)} pairs")
    print(f"  ID mapping example: {aligned_src[0]} -> {aligned_tgt[0]}")
    print(f"                      {aligned_src[1]} -> {aligned_tgt[1]}")


if __name__ == "__main__":
    generate()
