import argparse
import json
from pathlib import Path


def item_id(item):
    return str(item.get("id") or item.get("ocr_id") or item.get("transcription") + str(item.get("bbox")))


def clean_text(text):
    return (text or "").strip()


def clean_key(text):
    return clean_text(text).rstrip("：:")


def load_ser_items(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("ocr_info", [])
    return data


def load_relations(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("relations", [])
    return data


def logic_row(item):
    logic = item.get("logic_box") or []
    if len(logic) >= 2:
        return int(logic[0])
    return None


def logic_col(item):
    logic = item.get("logic_box") or []
    if len(logic) >= 4:
        return int(logic[2])
    return None


def row_group_key(item):
    return (
        int(item.get("page_idx", 0)),
        str(item.get("table_id", "")),
        logic_row(item),
    )


def sort_item_in_row(item):
    col = logic_col(item)
    bbox = item.get("bbox", [0, 0, 0, 0])
    return (
        col if col is not None else 999999,
        bbox[0],
        bbox[1],
    )


def group_rows(items):
    grouped = {}

    for item in items:
        row = logic_row(item)
        if row is None:
            continue

        key = row_group_key(item)
        grouped.setdefault(key, []).append(item)

    rows = []
    for key, row_items in grouped.items():
        page_idx, table_id, row_idx = key
        row_items.sort(key=sort_item_in_row)
        rows.append({
            "page_idx": page_idx,
            "table_id": table_id,
            "row_idx": row_idx,
            "items": row_items,
        })

    rows.sort(key=lambda r: (r["page_idx"], r["table_id"], r["row_idx"]))
    return rows


def pred_ratio(row_items, pred):
    if not row_items:
        return 0.0
    return sum(1 for x in row_items if x.get("pred") == pred) / len(row_items)


def short_text_ratio(row_items):
    if not row_items:
        return 0.0

    count = 0
    for item in row_items:
        text = clean_text(item.get("transcription", ""))
        if 1 <= len(text) <= 8:
            count += 1

    return count / len(row_items)


def looks_like_subtable_header(row_items):
    if len(row_items) < 3:
        return False

    question_ratio = pred_ratio(row_items, "QUESTION")
    header_ratio = pred_ratio(row_items, "HEADER")
    short_ratio = short_text_ratio(row_items)

    return (
        question_ratio >= 0.5
        or question_ratio + header_ratio >= 0.5
        or short_ratio >= 0.7
    )


def looks_like_subtable_data(row_items):
    if len(row_items) < 2:
        return False

    answer_ratio = pred_ratio(row_items, "ANSWER")
    question_ratio = pred_ratio(row_items, "QUESTION")

    return (
        answer_ratio >= 0.3
        or question_ratio <= 0.3
    )


def make_unique_field_keys(fields):
    seen = {}
    out = []

    for field in fields:
        key = clean_key(field)
        if not key:
            key = "字段"

        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            out.append(key)
        else:
            out.append(f"{key}_{seen[key]}")

    return out


def build_subtables(items):
    rows = group_rows(items)
    tables = []
    used_item_ids = set()

    i = 0
    while i < len(rows):
        header_row = rows[i]
        header_items = header_row["items"]

        if not looks_like_subtable_header(header_items):
            i += 1
            continue

        data_rows = []
        j = i + 1

        while j < len(rows):
            row = rows[j]

            if row["page_idx"] != header_row["page_idx"] or row["table_id"] != header_row["table_id"]:
                break

            row_items = row["items"]

            if looks_like_subtable_header(row_items) and data_rows:
                break

            if not looks_like_subtable_data(row_items):
                break

            data_rows.append(row)
            j += 1

        if len(data_rows) < 2:
            i += 1
            continue

        field_names = [clean_text(x.get("transcription", "")) for x in header_items]
        unique_keys = make_unique_field_keys(field_names)

        fields = [
            {
                "key": unique_keys[idx],
                "name": field_names[idx],
                "index": idx,
            }
            for idx in range(len(field_names))
        ]

        records = []
        record_objects = []

        for row in data_rows:
            values = [clean_text(x.get("transcription", "")) for x in row["items"]]
            records.append(values)

            obj = {}
            for idx, key in enumerate(unique_keys):
                obj[key] = values[idx] if idx < len(values) else ""
            record_objects.append(obj)

            for item in row["items"]:
                used_item_ids.add(item_id(item))

        for item in header_items:
            used_item_ids.add(item_id(item))

        tables.append({
            "page_idx": header_row["page_idx"],
            "table_id": header_row["table_id"],
            "header_row": header_row["row_idx"],
            "fields": fields,
            "records": records,
            "record_objects": record_objects,
        })

        i = j

    return tables, used_item_ids


def build_titles(items, table_item_ids):
    headers = []

    for item in items:
        if item_id(item) in table_item_ids:
            continue
        if item.get("pred") == "HEADER":
            text = clean_text(item.get("transcription", ""))
            if text:
                headers.append(text)

    return " ".join(headers[:3])


def build_attributes(relations, table_item_ids):
    attributes = {}
    relation_items = []

    for rel in relations:
        question = rel.get("question", {})
        answer = rel.get("answer", {})

        question_id = rel.get("question_id") or item_id(question)
        answer_id = rel.get("answer_id") or item_id(answer)

        if question_id in table_item_ids or answer_id in table_item_ids:
            continue

        key = clean_key(rel.get("key") or question.get("transcription", ""))
        value = clean_text(rel.get("value") or answer.get("transcription", ""))

        if not key or not value:
            continue

        if key in attributes and attributes[key] != value:
            if isinstance(attributes[key], list):
                attributes[key].append(value)
            else:
                attributes[key] = [attributes[key], value]
        else:
            attributes[key] = value

        relation_items.append(rel)

    return attributes, relation_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ser_json", required=True)
    parser.add_argument("--re_json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    items = load_ser_items(args.ser_json)
    relations = load_relations(args.re_json)

    tables, table_item_ids = build_subtables(items)
    attributes, filtered_relations = build_attributes(relations, table_item_ids)
    titles = build_titles(items, table_item_ids)

    payload = {
        "attributes": attributes,
        "titles": titles,
        "tables": tables,
        "items": items,
        "relations": filtered_relations,
        "excluded_table_item_ids": sorted(table_item_ids),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"saved: {output}")
    print(f"attributes: {len(attributes)}")
    print(f"tables: {len(tables)}")
    print(f"excluded table items: {len(table_item_ids)}")


if __name__ == "__main__":
    main()