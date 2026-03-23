import argparse
import os
import json
import re
from typing import Any, Dict, List, Optional
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:6006/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "qwen3")
INPUT_FILE = os.getenv("INPUT_FILE", "./data/source_content.csv")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "./data/source_profile.csv")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

PROFILE_SCHEMA_DESC = {
    "profile_summary": "用户画像总结，1到3句话，只基于文本证据，不猜测任何敏感属性。",
    "topic_labels": "内容主题标签，可有多个类别，但每个类别最多2个标签。建议类别如 兴趣话题/关注内容/日常场景。",
    "lifestyle_labels": "生活方式标签，可有多个类别，但每个类别最多2个标签。建议类别如 作息习惯/娱乐方式/消费饮食。",
    "language_style_labels": "表达风格标签，可有多个类别，但每个类别最多2个标签。建议类别如 表达方式/情绪表达/文本特征。",
    "emotion_tone": "整体情绪基调，只能从 积极 / 中性 / 混合 中选一个。",
    "content_language": "主要语言类型，如 中文 / 英文 / 中英混合。",
    "confidence": "画像置信度，只能从 高 / 中 / 低 中选一个。"
}


SYS_PROMPT = f"""
你是一名社交平台用户画像分析专家。
你的任务是根据某个用户的历史发文内容，生成“非敏感、可解释、结构化”的用户画像 JSON。

【分析原则】
1. 只能依据输入文本本身进行分析，不允许臆测。
2. 严禁推断或输出以下敏感属性：性别、年龄、民族、宗教、政治立场、疾病健康、收入、精确地理位置、性取向等。
3. 标签要稳定、抽象适中，优先提取长期偏好、日常行为倾向、表达方式。
4. 若内容较少或重复严重，应降低 confidence，且总结保持克制。
5. 输出必须是合法 JSON，不要输出 markdown，不要输出额外解释。
6. 每个标签字段都要设计为“对象”形式：可以包含多个类别，但每个类别下最多只能放 2 个标签。
7. 标签尽量简洁，使用短词或短语，不要写长句。

【输出字段说明】
{json.dumps(PROFILE_SCHEMA_DESC, ensure_ascii=False, indent=2)}

【严格输出格式】
{{
  "profile_summary": "",
  "topic_labels": {{
    "类别1": ["标签1", "标签2"],
    "类别2": ["标签1"]
  }},
  "lifestyle_labels": {{
    "类别1": ["标签1", "标签2"]
  }},
  "language_style_labels": {{
    "类别1": ["标签1", "标签2"]
  }},
  "emotion_tone": "",
  "content_language": "",
  "confidence": ""
}}
""".strip()


USER_PROMPT = """
请根据以下用户历史内容生成用户画像。

账号ID：{account_id}
群组ID：{room_id}
发文内容：
{content}

请严格输出 JSON 对象。
""".strip()


def call_llm(account_id: str, room_id: str, content: str, model_name: str) -> str:
    prompt = USER_PROMPT.format(account_id=account_id, room_id=room_id, content=content)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def clean_json_response(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None


def normalize_grouped_labels(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {}

    result: Dict[str, List[str]] = {}
    for k, v in value.items():
        key = str(k).strip()
        if not key:
            continue

        if isinstance(v, list):
            vals = [str(x).strip() for x in v if str(x).strip()]
        elif isinstance(v, str) and v.strip():
            vals = [x.strip() for x in re.split(r"[，,;；/|]", v) if x.strip()]
        else:
            vals = []

        dedup = []
        seen = set()
        for item in vals:
            if item not in seen:
                seen.add(item)
                dedup.append(item)

        if dedup:
            result[key] = dedup[:2]

    return result


def post_process_profile(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "profile_summary": "内容不足，暂无法稳定刻画该用户画像。",
            "topic_labels": {},
            "lifestyle_labels": {},
            "language_style_labels": {},
            "emotion_tone": "中性",
            "content_language": "未知",
            "confidence": "低",
        }

    emotion_tone = str(data.get("emotion_tone", "中性")).strip() or "中性"
    if emotion_tone not in {"积极", "中性", "混合"}:
        emotion_tone = "中性"

    confidence = str(data.get("confidence", "低")).strip() or "低"
    if confidence not in {"高", "中", "低"}:
        confidence = "低"

    return {
        "profile_summary": str(data.get("profile_summary", "")).strip() or "内容不足，暂无法稳定刻画该用户画像。",
        "topic_labels": normalize_grouped_labels(data.get("topic_labels")),
        "lifestyle_labels": normalize_grouped_labels(data.get("lifestyle_labels")),
        "language_style_labels": normalize_grouped_labels(data.get("language_style_labels")),
        "emotion_tone": emotion_tone,
        "content_language": str(data.get("content_language", "未知")).strip() or "未知",
        "confidence": confidence,
    }


def load_user_content(csv_file: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding="utf-8-sig")

    required_cols = {"account_id", "room_id", "content"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 缺少必要字段: {required_cols}，当前字段为: {list(df.columns)}")

    df = df.dropna(subset=["account_id", "content"]).copy()
    df["account_id"] = df["account_id"].astype(str).str.strip()
    df["room_id"] = df["room_id"].fillna("").astype(str).str.strip()
    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"] != ""]
    return df


def build_profiles(input_csv: str, output_csv: str, model_name: str) -> None:
    print(f"[INFO] 读取数据: {input_csv}")
    df = load_user_content(input_csv)
    print(f"[INFO] 待处理用户数: {len(df)}")
    # 测试
    # df = df.head(5).copy()
    results = []
    success_count = 0
    fail_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating profiles"):
        account_id = row["account_id"]
        room_id = row["room_id"]
        content = row["content"]

        try:
            raw = call_llm(account_id, room_id, content, model_name)
            parsed = clean_json_response(raw)
            profile = post_process_profile(parsed)
            success_count += 1
        except Exception as e:
            print(f"[WARN] 用户 {account_id} 调用失败: {e}")
            profile = post_process_profile(None)
            fail_count += 1

        results.append({
            "account_id": account_id,
            "room_id": room_id,
            "content": content,
            "profile_json": json.dumps(profile, ensure_ascii=False),
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n[INFO] 处理完成")
    print(f"[INFO] 成功生成画像: {success_count}")
    print(f"[INFO] 失败回退数量: {fail_count}")
    print(f"[INFO] 输出文件: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据用户 content 调用大模型生成画像 JSON")
    parser.add_argument("input_file", help="输入 CSV 文件路径")
    parser.add_argument("output_file", help="输出 CSV 文件路径")
    parser.add_argument("--model", default="qwen3", help="模型名称")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_profiles(args.input_file, args.output_file, args.model)
