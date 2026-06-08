"""
作用：把融合后的 RE 结果转成接口最终结构：

{
  "attribute": {},
  "titles": "",
  "subtables": []
}
"""

import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def text_of(item):
    return (item.get("transcription") or item.get("text") or "").strip()


def pred_of(item):
    return item.get("final_pred") or item.get("pred") or item.get("label") or ""


def add_attribute(attributes, key, value):
    key = (key or "").strip()
    value = (value or "").strip()
    if not key or not value:
        return

    if key not in attributes:
        attributes[key] = value
        return

    old = attributes[key]
    if old == value:
        return

    if isinstance(old, list):
        if value not in old:
            old.append(value)
        return

    attributes[key] = [old, value]


def collect_titles(final_root):
    titles = []
    seen = set()

    for path in sorted(Path(final_root).glob("logical_table_*_ser_merged.json")):
        data = read_json(path)
        for item in data.get("items", []) or []:
            if pred_of(item) != "HEADER":
                continue
            text = text_of(item)
            if not text or text in seen:
                continue
            seen.add(text)
            titles.append(text)

    return "；".join(titles)


def collect_attributes(final_root):
    attributes = {}
    relation_count = 0

    for path in sorted(Path(final_root).glob("logical_table_*_ser_merged.json")):
        data = read_json(path)
        relations = data.get("final_re", {}).get("final_relations") or []

        for rel in relations:
            key = rel.get("question_text") or rel.get("key") or ""
            value = rel.get("answer_text") or rel.get("value") or ""
            before = json.dumps(attributes, ensure_ascii=False, sort_keys=True)
            add_attribute(attributes, key, value)
            after = json.dumps(attributes, ensure_ascii=False, sort_keys=True)
            if before != after:
                relation_count += 1

    return attributes, relation_count


def load_subtable_export(export):
    headers_path = export.get("headers_json")
    rows_path = export.get("rows_json")

    if not headers_path or not rows_path:
        return None

    headers_file = Path(headers_path)
    rows_file = Path(rows_path)
    if not headers_file.exists() or not rows_file.exists():
        return None

    headers = read_json(headers_file)
    rows = read_json(rows_file)

    fields = {}
    for index, header in enumerate(headers):
        field_index = str(header.get("index", index))
        fields[field_index] = (header.get("header_text") or "").strip()

    records = []
    for row in rows:
        record = {}
        for cell in row.get("cells", []) or []:
            cell_index = str(cell.get("index", len(record)))
            record[cell_index] = (cell.get("value_text") or "").strip()
        records.append(record)

    return {
        "fields": fields,
        "records": records,
    }


def collect_subtables(subtables_root):
    manifest_path = Path(subtables_root) / "manifest.json"
    if not manifest_path.exists():
        return [], {"subtable_manifest": str(manifest_path), "missing": True}

    manifest = read_json(manifest_path)
    subtables = []
    skipped = []

    for export in manifest.get("exports", []) or []:
        subtable = load_subtable_export(export)
        if subtable is None:
            skipped.append(export)
            continue
        subtables.append(subtable)

    audit = {
        "subtable_manifest": str(manifest_path),
        "export_count": manifest.get("export_count", len(manifest.get("exports", []) or [])),
        "loaded_subtable_count": len(subtables),
        "skipped_subtable_count": len(skipped),
        "skipped": skipped,
    }
    return subtables, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_root", required=True)
    parser.add_argument("--subtables_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit_output", default=None)
    args = parser.parse_args()

    attributes, relation_count = collect_attributes(args.final_root)
    titles = collect_titles(args.final_root)
    subtables, subtable_audit = collect_subtables(args.subtables_root)

    result = {
        "attribute": attributes,
        "titles": titles,
        "subtables": subtables,
    }

    audit = {
        "script": "build_kie_final_payload.py",
        "final_root": str(Path(args.final_root)),
        "subtables_root": str(Path(args.subtables_root)),
        "attribute_count": len(attributes),
        "relation_count_added_to_attribute": relation_count,
        "title_count": 0 if not titles else len(titles.split("；")),
        "subtable_count": len(subtables),
        "subtables": subtable_audit,
    }

    write_json(args.output, result)
    if args.audit_output:
        write_json(args.audit_output, audit)

    print(f"result={args.output}")
    if args.audit_output:
        print(f"audit={args.audit_output}")


if __name__ == "__main__":
    main()