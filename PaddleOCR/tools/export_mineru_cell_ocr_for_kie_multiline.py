"""Export MinerU table-cell OCR items to PaddleOCR KIE ocr_info format.

Based on export_mineru_cell_ocr_for_kie.py, with conservative multiline merge
inside the same table cell before key/value splitting.
"""

import argparse
import json
import re
from pathlib import Path


def bbox_to_points(bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def box_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def points_to_int(points):
    return [[int(x), int(y)] for x, y in points]


def union_bbox(bboxes):
    xs1, ys1, xs2, ys2 = [], [], [], []
    for b in bboxes:
        xs1.append(b[0])
        ys1.append(b[1])
        xs2.append(b[2])
        ys2.append(b[3])
    return [int(min(xs1)), int(min(ys1)), int(max(xs2)), int(max(ys2))]


def bbox_height(b):
    return max(1, b[3] - b[1])


def bbox_width(b):
    return max(1, b[2] - b[0])


def center_y(b):
    return (b[1] + b[3]) / 2.0


def x_overlap_ratio(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return overlap / float(max(1, min(bbox_width(a), bbox_width(b))))


def normalize_text(text):
    return (text or "").strip()


def has_colon(text):
    return "：" in text or ":" in text


def has_checkbox_or_option(text):
    markers = ["□", "☑", "☒", "√", "✓", "✔", "无异常", "异常", "是否", "是", "否"]
    return any(m in text for m in markers)


def looks_like_section_title(text):
    text = normalize_text(text)
    return bool(re.match(r"^([一二三四五六七八九十]+|\d+)[、.．]\s*\S{1,30}$", text))


def looks_like_table_value(text):
    text = normalize_text(text)
    if not text:
        return True
    patterns = [
        r"^\d{1,2}[.:：]\d{2}$",
        r"^\d{1,4}([./-]\d{1,2}){1,2}$",
        r"^\d+(\.\d+)?\s*(℃|%|%RH|RH|台|件)?$",
        r"^[\dA-Za-z_.:/：-]{1,8}$",
    ]
    return any(re.match(p, text) for p in patterns)


def is_merge_blocker(text):
    text = normalize_text(text)
    if not text:
        return True
    if has_colon(text):
        return True
    if has_checkbox_or_option(text):
        return True
    if looks_like_section_title(text):
        return True
    return False


def can_merge_text(a_text, b_text):
    a_text = normalize_text(a_text)
    b_text = normalize_text(b_text)

    if is_merge_blocker(a_text) or is_merge_blocker(b_text):
        return False

    # Avoid merging dense record values such as date/time/temp rows.
    if looks_like_table_value(a_text) and looks_like_table_value(b_text):
        return False

    return True


def is_vertical_continuation(a, b):
    ab = a["bbox"]
    bb = b["bbox"]

    if center_y(bb) <= center_y(ab):
        return False

    ah = bbox_height(ab)
    bh = bbox_height(bb)
    avg_h = (ah + bh) / 2.0

    # Same visual line should not be vertically merged.
    if abs(center_y(ab) - center_y(bb)) <= max(4, 0.55 * avg_h):
        return False

    gap = bb[1] - ab[3]
    if gap < -0.35 * avg_h:
        return False
    if gap > max(12, 1.2 * avg_h):
        return False

    overlap_ok = x_overlap_ratio(ab, bb) >= 0.45
    left_align_ok = abs(ab[0] - bb[0]) <= max(12, 0.8 * avg_h)
    center_align_ok = abs((ab[0] + ab[2]) / 2.0 - (bb[0] + bb[2]) / 2.0) <= max(20, 1.5 * avg_h)

    return overlap_ok or left_align_ok or center_align_ok


def make_merged_item(items, separator=""):
    texts = [normalize_text(i.get("text", "")) for i in items if normalize_text(i.get("text", ""))]
    bboxes = [i["bbox"] for i in items]
    first = dict(items[0])
    first["text"] = separator.join(texts)
    first["bbox"] = union_bbox(bboxes)
    first["score"] = min(float(i.get("score", 1.0)) for i in items)
    first["ocr_id"] = "+".join(str(i.get("ocr_id", "")) for i in items if i.get("ocr_id", ""))
    first["source_ocr_ids"] = [i.get("ocr_id", "") for i in items]
    first["merged_from_count"] = len(items)
    return first


def sort_cell_items_for_merge(ocr_items):
    valid = []
    for idx, item in enumerate(ocr_items):
        text = normalize_text(item.get("text", ""))
        bbox = item.get("bbox", [])
        if not text or not bbox or len(bbox) != 4:
            continue
        copied = dict(item)
        copied["_source_order"] = idx
        copied["bbox"] = [int(v) for v in bbox]
        valid.append(copied)

    # Mostly follows visual order inside a cell, but source_order is preserved in output extras.
    return sorted(valid, key=lambda x: (x["bbox"][1], x["bbox"][0], x["_source_order"]))


def merge_multiline_items_in_cell(ocr_items, separator=""):
    items = sort_cell_items_for_merge(ocr_items)
    if len(items) <= 1:
        return items

    result = []
    used = [False] * len(items)

    for i, item in enumerate(items):
        if used[i]:
            continue

        chain = [item]
        used[i] = True
        cur = item

        while True:
            next_idx = None
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                cand = items[j]
                if not can_merge_text(cur.get("text", ""), cand.get("text", "")):
                    continue
                if not is_vertical_continuation(cur, cand):
                    continue
                next_idx = j
                break

            if next_idx is None:
                break

            used[next_idx] = True
            chain.append(items[next_idx])
            cur = items[next_idx]

        if len(chain) > 1:
            result.append(make_merged_item(chain, separator=separator))
        else:
            result.append(item)

    return sorted(result, key=lambda x: min(x.get("_source_order", 0), x.get("_source_order", 0)))


def split_key_value_item(text, bbox, score=1.0, extra=None):
    if extra is None:
        extra = {}

    points = bbox_to_points(bbox)

    if not text:
        return []

    split_chars = ["：", ":"]
    split_pos = -1
    for ch in split_chars:
        split_pos = text.find(ch)
        if split_pos > 0:
            break

    if split_pos <= 0 or split_pos >= len(text) - 1:
        return [{
            **extra,
            "transcription": text,
            "bbox": box_to_bbox(points),
            "points": points_to_int(points),
            "score": float(score),
        }]

    left_char = text[split_pos - 1]
    right_char = text[split_pos + 1]
    if left_char.isdigit() and right_char.isdigit():
        return [{
            **extra,
            "transcription": text,
            "bbox": box_to_bbox(points),
            "points": points_to_int(points),
            "score": float(score),
        }]

    key_text = text[:split_pos + 1].strip()
    value_text = text[split_pos + 1:].strip()

    if not key_text or not value_text:
        return [{
            **extra,
            "transcription": text,
            "bbox": box_to_bbox(points),
            "points": points_to_int(points),
            "score": float(score),
        }]

    if len(text) > 40:
        return [{
            **extra,
            "transcription": text,
            "bbox": box_to_bbox(points),
            "points": points_to_int(points),
            "score": float(score),
        }]

    x1, y1, x2, y2 = box_to_bbox(points)
    width = max(1, x2 - x1)

    key_len = max(1, len(key_text))
    value_len = max(1, len(value_text))
    key_ratio = key_len / float(key_len + value_len)

    split_x = int(x1 + width * key_ratio)
    split_x = max(x1 + 1, min(split_x, x2 - 1))

    key_points = [[x1, y1], [split_x, y1], [split_x, y2], [x1, y2]]
    value_points = [[split_x, y1], [x2, y1], [x2, y2], [split_x, y2]]

    return [
        {
            **extra,
            "transcription": key_text,
            "bbox": box_to_bbox(key_points),
            "points": points_to_int(key_points),
            "score": float(score),
        },
        {
            **extra,
            "transcription": value_text,
            "bbox": box_to_bbox(value_points),
            "points": points_to_int(value_points),
            "score": float(score),
        },
    ]


def iter_model_jsons(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*_model.json"))


def extract_tables(model_json):
    data = json.loads(Path(model_json).read_text(encoding="utf-8"))
    tables = []

    for page_idx, page in enumerate(data):
        for det_idx, det in enumerate(page.get("layout_dets", [])):
            cell_items = det.get("wired_cell_ocr_items")
            if not cell_items:
                continue

            tables.append({
                "page_idx": page_idx,
                "table_id": det.get("index", det_idx),
                "table_bbox": det.get("bbox", []),
                "html": det.get("html", ""),
                "cells": cell_items,
            })

    return tables


def make_output_name(model_json):
    name = Path(model_json).name
    if name.endswith("_model.json"):
        return name.replace("_model.json", "_ocr_info.json")
    return Path(model_json).stem + "_ocr_info.json"


def export_one(model_json, output_dir, merge_multiline=True, merge_separator=""):
    model_json = Path(model_json)
    tables = extract_tables(model_json)

    kie_ocr_info = []
    for table_idx, table in enumerate(tables):
        table_id = table.get("table_id", table_idx)
        page_idx = table.get("page_idx", 0)
        table_bbox = table.get("table_bbox", [])

        for cell in table["cells"]:
            cell_id = cell.get("cell_id", "")
            cell_bbox = cell.get("cell_bbox", [])
            logic_box = cell.get("logic_box", [])

            ocr_items = cell.get("ocr_items", [])
            if merge_multiline:
                ocr_items = merge_multiline_items_in_cell(
                    ocr_items,
                    separator=merge_separator,
                )

            for ocr in ocr_items:
                text = normalize_text(ocr.get("text", ""))
                bbox = ocr.get("bbox", [])
                if not text or not bbox or len(bbox) != 4:
                    continue

                extra = {
                    "id": ocr.get("ocr_id", ""),
                    "ocr_id": ocr.get("ocr_id", ""),
                    "cell_id": cell_id,
                    "table_id": table_id,
                    "page_idx": page_idx,
                    "cell_bbox": cell_bbox,
                    "logic_box": logic_box,
                    "table_bbox": table_bbox,
                    "source_ocr_ids": ocr.get("source_ocr_ids", [ocr.get("ocr_id", "")]),
                    "merged_from_count": ocr.get("merged_from_count", 1),
                }

                kie_ocr_info.extend(
                    split_key_value_item(
                        text=text,
                        bbox=bbox,
                        score=ocr.get("score", 1.0),
                        extra=extra,
                    )
                )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / make_output_name(model_json)
    output_path.write_text(
        json.dumps(kie_ocr_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    merged_count = sum(1 for x in kie_ocr_info if x.get("merged_from_count", 1) > 1)
    print(f"saved: {output_path}, count={len(kie_ocr_info)}, merged_items={merged_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="MinerU output dir or *_model.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--disable_multiline_merge", action="store_true")
    parser.add_argument("--merge_separator", default="")
    args = parser.parse_args()

    model_jsons = iter_model_jsons(args.input)
    if not model_jsons:
        raise RuntimeError(f"No *_model.json found: {args.input}")

    for model_json in model_jsons:
        export_one(
            model_json,
            args.output_dir,
            merge_multiline=not args.disable_multiline_merge,
            merge_separator=args.merge_separator,
        )


if __name__ == "__main__":
    main()