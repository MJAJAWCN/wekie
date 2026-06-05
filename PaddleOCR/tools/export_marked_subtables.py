"""
作用：只读取已经标记好的 subtable_marked 结果，把每个子表导出成：

*_headers.json：原封不动的表头列表，允许重复表头
*_rows.json：每行每格的 header/value 对应关系
*.csv：原封不动表头 CSV，比如 日期,时间,温度,记录,日期,时间...
"""

import argparse
import csv
import json
from pathlib import Path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cell_sort_key(cell, item_by_id):
    header = item_by_id.get(cell.get("header_id"), {})
    lb = header.get("logic_box") or [999999, 999999, 999999, 999999]
    return (lb[2], lb[3], cell.get("header_id") or "")


def build_subtable_export(subtable, item_by_id):
    rows = subtable.get("rows") or []
    sorted_rows = []

    for row in rows:
        cells = list(row.get("cells") or [])
        cells.sort(key=lambda c: cell_sort_key(c, item_by_id))
        sorted_rows.append((row, cells))

    header_slots = []
    for _, cells in sorted_rows:
        for idx, cell in enumerate(cells):
            while len(header_slots) <= idx:
                header_slots.append(None)

            if header_slots[idx] is None:
                header_id = cell.get("header_id")
                header_item = item_by_id.get(header_id, {})
                header_slots[idx] = {
                    "index": idx,
                    "header_id": header_id,
                    "header_text": cell.get("header_text") or "",
                    "header_logic_box": header_item.get("logic_box"),
                    "header_bbox": header_item.get("logical_bbox") or header_item.get("bbox"),
                }

    header_slots = [
        slot or {
            "index": idx,
            "header_id": None,
            "header_text": "",
            "header_logic_box": None,
            "header_bbox": None,
        }
        for idx, slot in enumerate(header_slots)
    ]

    exported_rows = []
    csv_rows = []

    for row, cells in sorted_rows:
        row_cells = []
        csv_values = []

        for idx, header in enumerate(header_slots):
            cell = cells[idx] if idx < len(cells) else {}
            value_item_ids = cell.get("value_item_ids") or []
            value_items = [item_by_id[x] for x in value_item_ids if x in item_by_id]

            value_text = cell.get("value_text") or ""
            csv_values.append(value_text)

            row_cells.append({
                "index": idx,
                "header_id": header.get("header_id"),
                "header_text": header.get("header_text") or "",
                "value_text": value_text,
                "value_item_ids": value_item_ids,
                "value_logic_boxes": [x.get("logic_box") for x in value_items],
                "value_bboxes": [
                    x.get("logical_bbox") or x.get("bbox")
                    for x in value_items
                ],
            })

        exported_rows.append({
            "row_key": row.get("row_key"),
            "item_ids": row.get("item_ids") or [],
            "cells": row_cells,
        })
        csv_rows.append(csv_values)

    return header_slots, exported_rows, csv_rows


def export_file(input_file, output_dir):
    data = read_json(input_file)
    item_by_id = {x.get("id"): x for x in data.get("items", []) if x.get("id")}

    exports = []
    subtables = data.get("subtables") or []

    for idx, subtable in enumerate(subtables, start=1):
        subtable_id = subtable.get("subtable_id") or f"subtable_{idx:03d}"
        prefix = f"{input_file.stem}_{subtable_id}"

        headers, rows, csv_rows = build_subtable_export(subtable, item_by_id)

        headers_path = output_dir / f"{prefix}_headers.json"
        rows_path = output_dir / f"{prefix}_rows.json"
        csv_path = output_dir / f"{prefix}.csv"

        write_json(headers_path, headers)
        write_json(rows_path, rows)

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([x.get("header_text") or "" for x in headers])
            writer.writerows(csv_rows)

        exports.append({
            "input": str(input_file),
            "subtable_id": subtable_id,
            "header_count": len(headers),
            "row_count": len(rows),
            "headers_json": str(headers_path),
            "rows_json": str(rows_path),
            "csv": str(csv_path),
        })

    return exports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_exports = []
    for input_file in sorted(input_root.glob("logical_table_*_ser_merged.json")):
        all_exports.extend(export_file(input_file, output_root))

    write_json(output_root / "manifest.json", {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "export_count": len(all_exports),
        "exports": all_exports,
    })

    print(f"exported {len(all_exports)} subtables to {output_root}")


if __name__ == "__main__":
    main()
