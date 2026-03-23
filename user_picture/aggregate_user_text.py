﻿import pandas as pd
import sys

def aggregate_text_by_user(input_file, output_file):
    print(f"[INFO] 开始读取文件: {input_file}")
    df = pd.read_csv(input_file)

    print(f"[INFO] 原始总记录数: {len(df)}")

    # 只保留 text 类型
    df = df[df["type"] == "text"].copy()
    print(f"[INFO] text 类型记录数: {len(df)}")

    # 按时间排序
    df = df.sort_values(["account_id", "time"])

    # 按 account_id 聚合
    result = (
        df.groupby("account_id", as_index=False)
        .agg(
            room_id=("room_id", "first"),
            content=("msg", lambda x: "\n".join(x.astype(str)))
        )
    )

    # 统计信息
    user_count = result["account_id"].nunique()
    post_count = len(df)

    print(f"[INFO] 聚合后用户数: {user_count}")
    print(f"[INFO] 聚合前参与聚合的发文数: {post_count}")

    # 保存结果
    result.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[INFO] 聚合结果已保存到: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python aggregate_user_text.py 输入文件.csv 输出文件.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    aggregate_text_by_user(input_file, output_file)