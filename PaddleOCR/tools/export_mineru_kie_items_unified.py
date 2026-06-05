import argparse
import json
import re
from pathlib import Path


LABEL_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_（）()／/.\-]{1,18}[：:]")


def bbox_to_points(bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def union_bbox(items):
    xs1, ys1, xs2, ys2 = [], [], [], []
    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        xs1.append(x1)
        ys1.append(y1)
        xs2.append(x2)
        ys2.append(y2)
    return [int(min(xs1)), int(min(ys1)), int(max(xs2)), int(max(ys2))]


def make_item(text, bbox, score=1.0, extra=None):
    extra = extra or {}
    return {
        **extra,
        "transcription": text.strip(),
        "bbox": [int(v) for v in bbox],
        "points": bbox_to_points(bbox),
        "score": float(score),
    }


def is_checkbox_text(text):
    markers = ["□", "☑", "☒", "√", "无异常", "异常"]
    return any(s in text for s in markers)


def valid_colon(text, pos):
    if pos <= 0 or pos >= len(text) - 1:
        return False
    if text[pos - 1].isdigit() and text[pos + 1].isdigit():
        return False
    return True


def find_label_matches(text):
    matches = []
    for match in LABEL_RE.finditer(text):
        colon_pos = match.end() - 1
        if valid_colon(text, colon_pos):
            matches.append(match)
    return matches


def split_by_generic_labels(text, bbox, score=1.0, extra=None):
    text = (text or "").strip()
    if not text:
        return []

    if is_checkbox_text(text):
        return [make_item(text, bbox, score, extra)]

    matches = find_label_matches(text)
    if not matches:
        return [make_item(text, bbox, score, extra)]

    if len(matches) == 1 and matches[0].start() > 3:
        return [make_item(text, bbox, score, extra)]

    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(1, x2 - x1)
    total_len = max(1, len(text))

    out = []
    for idx, match in enumerate(matches):
        q_start = match.start()
        q_end = match.end()
        a_start = q_end
        a_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        q_text = text[q_start:q_end].strip()
        a_text = text[a_start:a_end].strip()

        qx1 = int(x1 + width * q_start / total_len)
        qx2 = int(x1 + width * q_end / total_len)
        ax1 = qx2
        ax2 = int(x1 + width * a_end / total_len)

        qx2 = max(qx1 + 1, min(qx2, x2))
        ax2 = max(ax1 + 1, min(ax2, x2))

        q_extra = dict(extra or {})
        q_extra["split_role"] = "question_candidate"
        out.append(make_item(q_text, [qx1, y1, qx2, y2], score, q_extra))

        if a_text:
            a_extra = dict(extra or {})
            a_extra["split_role"] = "answer_candidate"
            out.append(make_item(a_text, [ax1, y1, ax2, y2], score, a_extra))

    return out or [make_item(text, bbox, score, extra)]


def should_merge_cell_ocr_items(items):
    if len(items) <= 1:
        return False

    texts = [i.get("text", "").strip() for i in items if i.get("text", "").strip()]
    if not texts:
        return False

    joined = "".join(texts)

    if is_checkbox_text(joined):
        return False

    if any(find_label_matches(t) for t in texts):
        return False

    return True


def normalize_cell_items(cell):
    raw = []
    for ocr in cell.get("ocr_items", []):
        text = ocr.get("text", "").strip()
        bbox = ocr.get("bbox", [])
        if not text or len(bbox) != 4:
            continue

        raw.append({
            "ocr_id": ocr.get("ocr_id", ""),
            "text": text,
            "bbox": [int(v) for v in bbox],
            "score": float(ocr.get("score", 1.0)),
        })

    raw.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

    if should_merge_cell_ocr_items(raw):
        return [{
            "ocr_id": raw[0]["ocr_id"],
            "text": "".join(i["text"] for i in raw),
            "bbox": union_bbox(raw),
            "score": sum(i["score"] for i in raw) / max(1, len(raw)),
            "merged_from": [i["ocr_id"] for i in raw],
        }]

    return raw


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
            cells = det.get("wired_cell_ocr_items")
            if not cells:
                continue

            tables.append({
                "page_idx": page_idx,
                "table_id": det.get("index", det_idx),
                "table_bbox": det.get("bbox", []),
                "cells": cells,
            })

    return tables


def make_output_name(model_json):
    name = Path(model_json).name
    if name.endswith("_model.json"):
        return name.replace("_model.json", "_ocr_info.json")
    return Path(model_json).stem + "_ocr_info.json"


def export_one(model_json, output_dir):
    tables = extract_tables(model_json)
    kie_items = []

    for table_idx, table in enumerate(tables):
        for cell in table["cells"]:
            base_extra = {
                "cell_id": cell.get("cell_id", ""),
                "table_id": table.get("table_id", table_idx),
                "page_idx": table.get("page_idx", 0),
                "cell_bbox": cell.get("cell_bbox", []),
                "logic_box": cell.get("logic_box", []),
                "table_bbox": table.get("table_bbox", []),
            }

            normalized = normalize_cell_items(cell)

            for local_idx, ocr in enumerate(normalized):
                extra = dict(base_extra)
                raw_id = ocr.get("ocr_id") or f'{extra["cell_id"]}_o{local_idx}'
                extra["id"] = raw_id
                extra["ocr_id"] = raw_id

                if "merged_from" in ocr:
                    extra["merged_from"] = ocr["merged_from"]

                parts = split_by_generic_labels(
                    text=ocr["text"],
                    bbox=ocr["bbox"],
                    score=ocr.get("score", 1.0),
                    extra=extra,
                )

                for part_idx, part in enumerate(parts):
                    if len(parts) > 1:
                        part["id"] = f"{raw_id}_s{part_idx}"
                        part["ocr_id"] = part["id"]
                        part["source_ocr_id"] = raw_id
                    kie_items.append(part)

    kie_items.sort(key=lambda x: (
        int(x.get("page_idx", 0)),
        x["bbox"][1],
        x["bbox"][0],
    ))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / make_output_name(model_json)
    output_path.write_text(
        json.dumps(kie_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"saved: {output_path}, count={len(kie_items)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    model_jsons = iter_model_jsons(args.input)
    if not model_jsons:
        raise RuntimeError(f"No *_model.json found: {args.input}")

    for model_json in model_jsons:
        export_one(model_json, args.output_dir)


if __name__ == "__main__":
    main()