"""
读取合并后的 logical_table_xxx_ser_merged.json
-> 找值格里的 无资料
-> 把无资料修成 ANSWER
-> 找对应 key：先同一行左侧，再上一行正上方
-> 把 key 保护成 QUESTION
-> 输出到新文件夹，不覆盖原合并结果
"""
"""
无资料值格:
  final_pred = ANSWER

对应 key:
  先找同 row 左侧字段格
  找不到，再找上一 row 正上方字段格
  找到就 final_pred = QUESTION
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


PRED_ID = {
    "O": 0,
    "QUESTION": 1,
    "ANSWER": 3,
    "HEADER": 5,
}

NO_DATA_VALUES = {
    "无资料",
    "暂无资料",
    "无数据",
    "暂无数据",
    "未提供",
    "未见资料",
}


def normalize_text(text):
    text = text or ""
    text = re.sub(r"\s+", "", str(text))
    return text.strip("。；;：:")


def is_no_data_text(text):
    return normalize_text(text) in NO_DATA_VALUES


def item_text(item):
    return item.get("original_transcription") or item.get("transcription") or ""


def logic_box(item):
    lb = item.get("logic_box")
    if not lb or len(lb) != 4:
        return None
    return [int(x) for x in lb]


def is_full_span_cell(item):
    lb = logic_box(item)
    if not lb:
        return False
    _, _, col_start, col_end = lb
    return col_start == 0 and col_end >= 3


def is_value_cell(item):
    lb = logic_box(item)
    if not lb:
        return False

    row_start, row_end, col_start, col_end = lb

    if is_full_span_cell(item):
        return False

    if col_start in (1, 3):
        return True

    if col_start == 1 and col_end >= 3:
        return True

    return False


def set_pred(item, pred, reason):
    old_pred = item.get("final_pred")
    old_pred_id = item.get("final_pred_id")

    if old_pred == pred:
        item.setdefault("type_fix_reasons", [])
        if reason not in item["type_fix_reasons"]:
            item["type_fix_reasons"].append(reason)
        return False

    item["pred_before_type_fix"] = old_pred
    item["pred_id_before_type_fix"] = old_pred_id
    item["final_pred"] = pred
    item["final_pred_id"] = PRED_ID[pred]
    item.setdefault("type_fix_reasons", [])
    item["type_fix_reasons"].append(reason)
    return True


def y_overlap_ratio(a, b):
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]
    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    denom = max(1, min(ay2 - ay1, by2 - by1))
    return overlap / denom


def x_overlap_ratio(a, b):
    ax1, ax2 = a[0], a[2]
    bx1, bx2 = b[0], b[2]
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    denom = max(1, min(ax2 - ax1, bx2 - bx1))
    return overlap / denom


def bbox(item):
    b = item.get("logical_bbox") or item.get("virtual_bbox") or item.get("bbox")
    if not b or len(b) != 4:
        return None
    return [int(x) for x in b]


def same_fragment(a, b):
    return (
        a.get("group_id") == b.get("group_id")
        and a.get("fragment_index") == b.get("fragment_index")
    )


def candidate_sort_key(item):
    b = bbox(item) or [0, 0, 0, 0]
    return (b[1], b[0], str(item.get("id", "")))


def find_same_row_left_key(value_item, items):
    v_lb = logic_box(value_item)
    v_box = bbox(value_item)
    if not v_lb or not v_box:
        return None

    v_row_start, v_row_end, v_col_start, v_col_end = v_lb

    target_cols = []
    if v_col_start == 1:
        target_cols = [0]
    elif v_col_start == 3:
        target_cols = [2]
    else:
        return None

    candidates = []
    for item in items:
        if item is value_item:
            continue
        if not same_fragment(value_item, item):
            continue

        lb = logic_box(item)
        ibox = bbox(item)
        if not lb or not ibox:
            continue

        row_start, row_end, col_start, col_end = lb
        if row_start != v_row_start or row_end != v_row_end:
            continue
        if col_start not in target_cols:
            continue
        if ibox[2] > v_box[0]:
            continue
        if y_overlap_ratio(v_box, ibox) < 0.35:
            continue

        candidates.append(item)

    if not candidates:
        return None

    return sorted(candidates, key=candidate_sort_key)[-1]


def find_prev_row_above_key(value_item, items):
    v_lb = logic_box(value_item)
    v_box = bbox(value_item)
    if not v_lb or not v_box:
        return None

    v_row_start, _, v_col_start, v_col_end = v_lb
    prev_row = v_row_start - 1

    candidates = []
    for item in items:
        if item is value_item:
            continue
        if not same_fragment(value_item, item):
            continue

        lb = logic_box(item)
        ibox = bbox(item)
        if not lb or not ibox:
            continue

        row_start, row_end, col_start, col_end = lb
        if row_end != prev_row:
            continue

        # 正上方：logic col 有交集，或者 bbox x 方向明显重叠
        logic_col_overlap = not (col_end < v_col_start or col_start > v_col_end)
        bbox_x_overlap = x_overlap_ratio(v_box, ibox) >= 0.35
        if not (logic_col_overlap or bbox_x_overlap):
            continue

        pred = item.get("final_pred")
        text = normalize_text(item_text(item))

        # 上一行只保护像字段的东西，避免把长正文改成 question
        looks_like_short_key = 0 < len(text) <= 30
        if pred not in ("QUESTION", "HEADER", "O") and not looks_like_short_key:
            continue

        candidates.append(item)

    if not candidates:
        return None

    # 优先已有 QUESTION/HEADER，再按位置稳定选
    def key(item):
        pred = item.get("final_pred")
        priority = 2 if pred == "QUESTION" else 1 if pred == "HEADER" else 0
        ibox = bbox(item) or [0, 0, 0, 0]
        overlap = x_overlap_ratio(v_box, ibox)
        return (priority, overlap, -abs(ibox[0] - v_box[0]))

    return max(candidates, key=key)


def process_group(data):
    items = data.get("items", [])
    fixed_no_data = 0
    protected_key = 0

    for item in items:
        if not is_no_data_text(item_text(item)):
            continue
        if not is_value_cell(item):
            continue

        if set_pred(item, "ANSWER", "no_data_value_cell_to_answer"):
            fixed_no_data += 1

        key_item = find_same_row_left_key(item, items)
        key_reason = "no_data_same_row_left_key_to_question"

        if key_item is None:
            key_item = find_prev_row_above_key(item, items)
            key_reason = "no_data_prev_row_above_key_to_question"

        if key_item is not None:
            if set_pred(key_item, "QUESTION", key_reason):
                protected_key += 1
            item["protected_question_id"] = key_item.get("id")
            item["protected_question_reason"] = key_reason

    data["post_type_fix"] = {
        "name": "no_data_answer_and_key_protection",
        "fixed_no_data_to_answer": fixed_no_data,
        "protected_key_to_question": protected_key,
    }

    data["final_pred_counts_after_type_fix"] = dict(
        Counter(item.get("final_pred", "O") for item in items)
    )

    return data


def iter_group_files(input_root):
    input_root = Path(input_root)
    return sorted(input_root.glob("logical_table_*_ser_merged.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    group_summaries = []

    for in_path in iter_group_files(input_root):
        data = json.loads(in_path.read_text(encoding="utf-8"))
        data = process_group(data)

        out_path = output_root / in_path.name
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        group_summaries.append({
            "input": str(in_path),
            "output": str(out_path),
            "group_id": data.get("group_id"),
            "item_count": data.get("item_count"),
            "post_type_fix": data.get("post_type_fix"),
            "final_pred_counts_after_type_fix": data.get("final_pred_counts_after_type_fix"),
        })

        print(f"[OK] {in_path.name} -> {out_path}")
        print(json.dumps(data.get("post_type_fix"), ensure_ascii=False))

    manifest = {
        "source_input_root": str(input_root),
        "output_root": str(output_root),
        "groups": group_summaries,
    }

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[DONE]")
    print(f"input_root={input_root}")
    print(f"output_root={output_root}")
    print(f"manifest={output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()