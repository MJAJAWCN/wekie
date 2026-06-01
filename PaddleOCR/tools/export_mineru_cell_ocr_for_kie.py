"""Export MinerU table-cell OCR items to PaddleOCR KIE ocr_info format.

This script reads MinerU *_model.json files that contain wired_cell_ocr_items,
then flattens cell-level OCR items into LayoutXLM/KIE ocr_info JSON.
"""

import argparse
import json
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


def split_key_value_item(text, bbox, score=1.0, extra=None):
    """Same split logic as export_mineru_ocr_for_kie.py."""
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


def export_one(model_json, output_dir):
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

            for ocr in cell.get("ocr_items", []):
                text = ocr.get("text", "")
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

    print(f"saved: {output_path}, count={len(kie_ocr_info)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="MinerU output dir or *_model.json")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    model_jsons = iter_model_jsons(args.input)
    if not model_jsons:
        raise RuntimeError(f"No *_model.json found: {args.input}")

    for model_json in model_jsons:
        export_one(model_json, args.output_dir)


if __name__ == "__main__":
    main()