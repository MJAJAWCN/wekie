"""
读取最终 merged SER
-> 按 logic_box 聚成逻辑行
-> 找“多 QUESTION 表头行 + 连续多行 ANSWER 数据行”的子表
-> 给子表 item 标记 in_subtable / subtable_id
-> 输出 main_item_ids 和 subtables
-> 后续 RE 只用 main_item_ids
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def logic_box(item):
    lb = item.get("logic_box")
    if not lb or len(lb) != 4:
        return None
    return [int(x) for x in lb]


def bbox(item):
    b = item.get("logical_bbox") or item.get("virtual_bbox") or item.get("bbox")
    if not b or len(b) != 4:
        return [0, 0, 0, 0]
    return [int(x) for x in b]


def pred(item):
    return item.get("final_pred") or item.get("pred") or "O"


def text(item):
    return item.get("transcription", "")


def item_id(item):
    return item.get("id") or item.get("ocr_id")


def col_overlap(a, b):
    _, _, ac1, ac2 = a
    _, _, bc1, bc2 = b
    return not (ac2 < bc1 or bc2 < ac1)


def row_key(item):
    lb = logic_box(item)
    if not lb:
        return None
    rs, re, _, _ = lb
    return (
        item.get("group_id"),
        int(item.get("fragment_index", 0) or 0),
        rs,
        re,
    )


def build_rows(items):
    row_map = {}
    for item in items:
        key = row_key(item)
        if key is None:
            continue
        row_map.setdefault(key, []).append(item)

    rows = []
    for key, row_items in row_map.items():
        row_items.sort(key=lambda x: (bbox(x)[0], bbox(x)[1], str(item_id(x))))
        rows.append({
            "key": key,
            "group_id": key[0],
            "fragment_index": key[1],
            "row_start": key[2],
            "row_end": key[3],
            "items": row_items,
            "y1": min(bbox(x)[1] for x in row_items),
            "x1": min(bbox(x)[0] for x in row_items),
        })

    rows.sort(key=lambda r: (
        r["group_id"],
        r["fragment_index"],
        r["y1"],
        r["row_start"],
        r["x1"],
    ))
    return rows


def is_full_span_header_row(row):
    items = row["items"]
    if not items:
        return False

    for item in items:
        lb = logic_box(item)
        if not lb:
            continue
        _, _, cs, ce = lb
        if cs == 0 and ce >= 3 and pred(item) in ("HEADER", "QUESTION"):
            return True
    return False


def header_items(row):
    return [x for x in row["items"] if pred(x) == "QUESTION"]


def answer_items(row):
    return [x for x in row["items"] if pred(x) == "ANSWER"]


def is_subtable_header_row(row, min_header_questions):
    qs = header_items(row)
    ans = answer_items(row)

    if len(qs) < min_header_questions:
        return False

    # 普通 SDS 行常见 Q/A/Q/A，同一行有 ANSWER，不当子表表头。
    if ans:
        return False

    # section 标题不当子表表头。
    if is_full_span_header_row(row) and len(qs) <= 2:
        return False

    distinct_cols = set()
    for q in qs:
        lb = logic_box(q)
        if not lb:
            continue
        distinct_cols.add((lb[2], lb[3]))

    return len(distinct_cols) >= min_header_questions


def matched_answer_count(row, headers):
    count = 0
    for h in headers:
        h_lb = logic_box(h)
        if not h_lb:
            continue
        for a in answer_items(row):
            a_lb = logic_box(a)
            if a_lb and col_overlap(h_lb, a_lb):
                count += 1
                break
    return count


def is_data_row_for_header(row, headers, min_matched_cols):
    if is_full_span_header_row(row):
        return False

    if len(answer_items(row)) < min_matched_cols:
        return False

    return matched_answer_count(row, headers) >= min_matched_cols


def make_subtable(subtable_id, header_row, data_rows):
    headers = header_items(header_row)

    out_rows = []
    for row in data_rows:
        cells = []
        for h in headers:
            h_lb = logic_box(h)
            values = []
            for a in answer_items(row):
                a_lb = logic_box(a)
                if h_lb and a_lb and col_overlap(h_lb, a_lb):
                    values.append(a)

            cells.append({
                "header_id": item_id(h),
                "header_text": text(h),
                "value_item_ids": [item_id(x) for x in values],
                "value_text": " ".join(text(x) for x in values),
            })

        out_rows.append({
            "row_key": list(row["key"]),
            "item_ids": [item_id(x) for x in row["items"]],
            "cells": cells,
        })

    return {
        "subtable_id": subtable_id,
        "header_row_key": list(header_row["key"]),
        "header_item_ids": [item_id(x) for x in headers],
        "header_texts": [text(x) for x in headers],
        "data_row_keys": [list(r["key"]) for r in data_rows],
        "item_ids": (
            [item_id(x) for x in header_row["items"]]
            + [item_id(x) for r in data_rows for x in r["items"]]
        ),
        "rows": out_rows,
    }


def mark_items(subtable, id_to_item):
    for iid in subtable["header_item_ids"]:
        item = id_to_item.get(iid)
        if item:
            item["in_subtable"] = True
            item["subtable_id"] = subtable["subtable_id"]
            item["subtable_role"] = "header"

    for row in subtable["rows"]:
        for iid in row["item_ids"]:
            item = id_to_item.get(iid)
            if item and not item.get("subtable_role"):
                item["in_subtable"] = True
                item["subtable_id"] = subtable["subtable_id"]
                item["subtable_role"] = "cell"


def detect_subtables(items, min_header_questions, min_data_rows, min_matched_cols):
    rows = build_rows(items)
    id_to_item = {item_id(x): x for x in items if item_id(x)}
    subtables = []
    used_row_keys = set()

    i = 0
    while i < len(rows):
        row = rows[i]
        if row["key"] in used_row_keys:
            i += 1
            continue

        if not is_subtable_header_row(row, min_header_questions):
            i += 1
            continue

        headers = header_items(row)
        data_rows = []
        j = i + 1

        while j < len(rows):
            next_row = rows[j]

            if next_row["group_id"] != row["group_id"]:
                break

            if is_subtable_header_row(next_row, min_header_questions):
                break

            if is_data_row_for_header(next_row, headers, min_matched_cols):
                data_rows.append(next_row)
                j += 1
                continue

            break

        if len(data_rows) >= min_data_rows:
            subtable_id = f"subtable_{len(subtables) + 1:03d}"
            subtable = make_subtable(subtable_id, row, data_rows)
            subtables.append(subtable)
            mark_items(subtable, id_to_item)

            used_row_keys.add(row["key"])
            for r in data_rows:
                used_row_keys.add(r["key"])

            i = j
            continue

        i += 1

    return subtables


def process_group(data, args):
    items = data.get("items", [])
    subtables = detect_subtables(
        items,
        min_header_questions=args.min_header_questions,
        min_data_rows=args.min_data_rows,
        min_matched_cols=args.min_matched_cols,
    )

    subtable_item_ids = {
        iid
        for st in subtables
        for iid in st.get("item_ids", [])
    }

    main_item_ids = [
        item_id(x)
        for x in items
        if item_id(x) and item_id(x) not in subtable_item_ids
    ]

    data["subtable_detection"] = {
        "name": "logic_box_ser_subtable_detection",
        "subtable_count": len(subtables),
        "subtable_item_count": len(subtable_item_ids),
        "main_item_count": len(main_item_ids),
        "min_header_questions": args.min_header_questions,
        "min_data_rows": args.min_data_rows,
        "min_matched_cols": args.min_matched_cols,
    }
    data["subtables"] = subtables
    data["main_item_ids"] = main_item_ids
    data["subtable_item_ids"] = sorted(subtable_item_ids)

    data["final_pred_counts_after_subtable_mark"] = dict(
        Counter(x.get("final_pred", "O") for x in items)
    )

    return data


def iter_group_files(input_root):
    return sorted(Path(input_root).glob("logical_table_*_ser_merged.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--min_header_questions", type=int, default=2)
    parser.add_argument("--min_data_rows", type=int, default=2)
    parser.add_argument("--min_matched_cols", type=int, default=2)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []

    for in_path in iter_group_files(input_root):
        data = json.loads(in_path.read_text(encoding="utf-8"))
        data = process_group(data, args)

        out_path = output_root / in_path.name
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summaries.append({
            "input": str(in_path),
            "output": str(out_path),
            "group_id": data.get("group_id"),
            "item_count": data.get("item_count"),
            "subtable_detection": data.get("subtable_detection"),
        })

        print(f"[OK] {in_path.name}")
        print(json.dumps(data["subtable_detection"], ensure_ascii=False))

    manifest = {
        "source_input_root": str(input_root),
        "output_root": str(output_root),
        "groups": summaries,
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