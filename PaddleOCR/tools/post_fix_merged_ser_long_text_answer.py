"""
读取 no_data_key_fix 后的 merged SER 结果
-> 找 final_pred == O 的跨列长正文
-> 只找左侧紧邻一格，或上一行正上方同列一格的 QUESTION/HEADER
-> 找到后：
   长正文改成 ANSWER
   如果 key 是 HEADER，就改成 QUESTION，方便 RE 识别
-> 输出到新目录，不覆盖旧结果
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


def normalize_text(text):
    text = text or ""
    text = re.sub(r"\s+", "", str(text))
    return text.strip()


def logic_box(item):
    lb = item.get("logic_box")
    if not lb or len(lb) != 4:
        return None
    return [int(x) for x in lb]


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

# 让“上一行正上方”可以跨 fragment，但仍然只找 逻辑顺序里的前一行，不找前几行。

"""
改完后规则变成：

左侧 key:
  仍然只找同 fragment、同 row、左侧紧邻一格

上方 key:
  不再限制 same_fragment
  而是找同 group 里“排序后的上一行”
  仍然要求 logic_box col_start / col_end 完全一致
所以它能修：

fragment 2 最后一行：第十二部分：废弃处置
fragment 3 第一行：处置前应参阅...
"""
def group_id(item):
    return item.get("group_id")


def fragment_index(item):
    return int(item.get("fragment_index", 0) or 0)


def item_y1(item):
    b = bbox(item)
    if not b:
        return 0
    return int(b[1])


def item_x1(item):
    b = bbox(item)
    if not b:
        return 0
    return int(b[0])


def row_key(item):
    lb = logic_box(item)
    if not lb:
        return None
    row_start, row_end, _, _ = lb
    return (
        group_id(item),
        fragment_index(item),
        row_start,
        row_end,
    )


def build_ordered_rows(items):
    row_map = {}
    for item in items:
        lb = logic_box(item)
        if not lb:
            continue
        key = row_key(item)
        if key is None:
            continue
        row_map.setdefault(key, []).append(item)

    rows = []
    for key, row_items in row_map.items():
        rows.append({
            "key": key,
            "group_id": key[0],
            "fragment_index": key[1],
            "row_start": key[2],
            "row_end": key[3],
            "items": row_items,
            "y1": min(item_y1(x) for x in row_items),
            "x1": min(item_x1(x) for x in row_items),
        })

    rows.sort(key=lambda r: (
        r["group_id"],
        r["fragment_index"],
        r["y1"],
        r["row_start"],
        r["x1"],
    ))

    return rows


def find_previous_ordered_row(answer_item, ordered_rows):
    a_key = row_key(answer_item)
    if a_key is None:
        return None

    prev = None
    for row in ordered_rows:
        if row["key"] == a_key:
            return prev
        if row["group_id"] == group_id(answer_item):
            prev = row

    return None

def is_long_o_text(item, min_len=25):
    if item.get("final_pred") != "O":
        return False

    text = normalize_text(item.get("transcription", ""))
    if len(text) < min_len:
        return False

    lb = logic_box(item)
    if not lb:
        return False

    _, _, col_start, col_end = lb

    # 跨列正文格。这里不要求必须 0-3，因为 table 001 有 0-1 的两列逻辑表。
    return (col_end - col_start + 1) >= 2


def set_pred(item, pred, reason):
    old_pred = item.get("final_pred")
    old_pred_id = item.get("final_pred_id")

    if old_pred == pred:
        item.setdefault("type_fix_reasons", [])
        if reason not in item["type_fix_reasons"]:
            item["type_fix_reasons"].append(reason)
        return False

    item["pred_before_long_text_fix"] = old_pred
    item["pred_id_before_long_text_fix"] = old_pred_id
    item["final_pred"] = pred
    item["final_pred_id"] = PRED_ID[pred]
    item.setdefault("type_fix_reasons", [])
    item["type_fix_reasons"].append(reason)
    return True


# def find_left_adjacent_key(answer_item, items):
#     a_lb = logic_box(answer_item)
#     if not a_lb:
#         return None

#     a_row_start, a_row_end, a_col_start, _ = a_lb

#     candidates = []
#     for item in items:
#         if item is answer_item:
#             continue
#         if not same_fragment(answer_item, item):
#             continue
#         if item.get("final_pred") not in ("QUESTION", "HEADER"):
#             continue

#         lb = logic_box(item)
#         if not lb:
#             continue

#         row_start, row_end, _, col_end = lb

#         # 同一行，左侧紧邻一格/一列
#         if row_start != a_row_start or row_end != a_row_end:
#             continue
#         if col_end + 1 != a_col_start:
#             continue

#         candidates.append(item)

#     if not candidates:
#         return None

#     # 理论上只有一个；多了就取最靠右的
#     return max(candidates, key=lambda x: logic_box(x)[3])

# 把“找左边/上边 key”的逻辑从 bbox 对齐改成 logic_box 单元格关系，支持合并单元格。
def find_left_adjacent_key(answer_item, items):
    a_lb = logic_box(answer_item)
    if not a_lb:
        return None

    a_row_start, a_row_end, a_col_start, _ = a_lb

    candidates = []
    for item in items:
        if item is answer_item:
            continue
        if not same_fragment(answer_item, item):
            continue
        if item.get("final_pred") not in ("QUESTION", "HEADER"):
            continue

        lb = logic_box(item)
        if not lb:
            continue

        row_start, row_end, _, col_end = lb

        # 左边紧邻 cell：
        # 1. 列必须紧邻；
        # 2. 行可以完全相同，也可以是左侧合并单元格覆盖当前行。
        col_adjacent = col_end + 1 == a_col_start
        row_covers = row_start <= a_row_start and row_end >= a_row_end

        if not (col_adjacent and row_covers):
            continue

        candidates.append(item)

    if not candidates:
        return None

    # 如果同一个左侧 cell 里有多个 item，优先选非“第X部分：”的文本
    def score(item):
        text = normalize_text(item.get("transcription", ""))
        is_part_prefix = bool(re.match(r"^第.+部分[:：]?$", text))
        return (
            0 if is_part_prefix else 1,
            len(text),
        )

    return max(candidates, key=score)


# def find_above_same_cell_key(answer_item, items):
#     a_lb = logic_box(answer_item)
    # if not a_lb:
    #     return None

    # a_row_start, _, a_col_start, a_col_end = a_lb

    # candidates = []
    # for item in items:
    #     if item is answer_item:
    #         continue
    #     if not same_fragment(answer_item, item):
    #         continue
    #     if item.get("final_pred") not in ("QUESTION", "HEADER"):
    #         continue

    #     lb = logic_box(item)
    #     if not lb:
    #         continue

    #     row_start, row_end, col_start, col_end = lb

    #     # 只找上一行，且同列/同格完全对齐
    #     if row_end + 1 != a_row_start:
    #         continue
    #     if col_start != a_col_start or col_end != a_col_end:
    #         continue

    #     candidates.append(item)

    # if not candidates:
    #     return None

    # # 理论上只有一个；多了就取最靠下的
    # return max(candidates, key=lambda x: logic_box(x)[1])

# 下面这个函数改成了 find_above_same_cell_key，允许跨 fragment，但仍然只找逻辑顺序里的上一行，不找前几行。
def find_above_same_cell_key(answer_item, items, ordered_rows=None):
    a_lb = logic_box(answer_item)
    if not a_lb:
        return None

    _, _, a_col_start, a_col_end = a_lb

    if ordered_rows is None:
        ordered_rows = build_ordered_rows(items)

    prev_row = find_previous_ordered_row(answer_item, ordered_rows)
    if prev_row is None:
        return None

    candidates = []
    for item in prev_row["items"]:
        if item is answer_item:
            continue
        if group_id(item) != group_id(answer_item):
            continue
        if item.get("final_pred") not in ("QUESTION", "HEADER"):
            continue

        lb = logic_box(item)
        if not lb:
            continue

        _, _, col_start, col_end = lb

        # 上一逻辑行，且同列/同格完全对齐
        # if col_start != a_col_start or col_end != a_col_end:
        #     continue
        
        # 上一逻辑行的 cell 关系：
        # 1. 同列/同格完全对齐；
        # 2. 或者上一行是横向合并单元格，覆盖当前 answer 的列范围。
        same_cell = col_start == a_col_start and col_end == a_col_end
        covers_answer = col_start <= a_col_start and col_end >= a_col_end

        if not (same_cell or covers_answer):
            continue

        candidates.append(item)

    if not candidates:
        return None

    # 理论上只有一个；如果标题被冒号拆成两段，优先取文本部分，不优先取“第X部分：”
    def score(item):
        text = normalize_text(item.get("transcription", ""))
        is_part_prefix = bool(re.match(r"^第.+部分[:：]?$", text))
        return (
            0 if is_part_prefix else 1,
            len(text),
        )

    return max(candidates, key=score)


def protect_key_for_re(key_item, reason):
    if key_item.get("final_pred") == "HEADER":
        set_pred(key_item, "QUESTION", reason)
        key_item["long_text_key_was_header"] = True
        return True

    key_item.setdefault("type_fix_reasons", [])
    if reason not in key_item["type_fix_reasons"]:
        key_item["type_fix_reasons"].append(reason)
    return False


def process_group(data):
    items = data.get("items", [])
    ordered_rows = build_ordered_rows(items)
    fixed_long_text = 0
    header_to_question = 0

    for item in items:
        if not is_long_o_text(item):
            continue

        key_item = find_left_adjacent_key(item, items)
        key_reason = "long_text_left_adjacent_key"

        if key_item is None:
            # key_item = find_above_same_cell_key(item, items)
            key_item = find_above_same_cell_key(item, items, ordered_rows=ordered_rows)
            key_reason = "long_text_above_same_cell_key"

        if key_item is None:
            continue

        if set_pred(item, "ANSWER", "long_text_o_to_answer"):
            fixed_long_text += 1

        item["protected_question_id"] = key_item.get("id")
        item["protected_question_reason"] = key_reason

        if protect_key_for_re(key_item, key_reason + "_to_question_for_re"):
            header_to_question += 1

    data["post_long_text_fix"] = {
        "name": "long_text_o_answer_with_adjacent_key",
        "fixed_long_text_to_answer": fixed_long_text,
        "header_key_to_question": header_to_question,
    }

    data["final_pred_counts_after_long_text_fix"] = dict(
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
            "post_long_text_fix": data.get("post_long_text_fix"),
            "final_pred_counts_after_long_text_fix": data.get("final_pred_counts_after_long_text_fix"),
        })

        print(f"[OK] {in_path.name} -> {out_path}")
        print(json.dumps(data.get("post_long_text_fix"), ensure_ascii=False))

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
