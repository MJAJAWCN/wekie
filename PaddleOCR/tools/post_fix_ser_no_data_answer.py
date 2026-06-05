"""处理比如SER之后把“无资料”标记为other这种，加一个规则处理层，把这些标记为answer"""


import argparse
import json
import re
from pathlib import Path


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
    text = re.sub(r"\s+", "", text)
    text = text.strip("。；;，,：:")
    return text


def is_no_data_value(text):
    return normalize_text(text) in NO_DATA_VALUES


def logic_box_says_value_cell(item):
    logic_box = item.get("logic_box")
    if not logic_box or len(logic_box) != 4:
        return False

    _, _, col_start, _ = logic_box

    # In these MinerU table cells, col 0 is usually the field/key column.
    # A no-data marker in col 1+ is very likely a value.
    return int(col_start) > 0


def y_overlap_ratio(a, b):
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]
    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    denom = max(1, min(ay2 - ay1, by2 - by1))
    return overlap / denom


def nearby_question_says_value_cell(item, all_items):
    bbox = item.get("bbox")
    if not bbox or len(bbox) != 4:
        return False

    x1, y1, x2, y2 = bbox
    cy = (y1 + y2) / 2.0

    for other in all_items:
        if other is item:
            continue
        if other.get("pred") != "QUESTION":
            continue

        obox = other.get("bbox")
        if not obox or len(obox) != 4:
            continue

        ox1, oy1, ox2, oy2 = obox
        ocy = (oy1 + oy2) / 2.0

        same_row_left_key = ox2 <= x1 and y_overlap_ratio(bbox, obox) >= 0.45
        nearby_above_key = abs(ocy - cy) < 120 and ox1 <= x1 and ox2 <= x2

        if same_row_left_key or nearby_above_key:
            return True

    return False


def should_fix_to_answer(item, all_items):
    if not is_no_data_value(item.get("transcription", "")):
        return False

    pred = item.get("pred")
    pred_id = item.get("pred_id")

    if pred not in ("O", "OTHER") and pred_id != 0:
        return False

    return logic_box_says_value_cell(item) or nearby_question_says_value_cell(item, all_items)


def process_infer(infer_path, output_path):
    infer_path = Path(infer_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fixed_count = 0
    image_count = 0
    item_count = 0

    with infer_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue

            if "\t" not in line:
                fout.write(line + "\n")
                continue

            image_path, payload = line.split("\t", 1)
            data = json.loads(payload)
            ocr_info = data.get("ocr_info", [])

            for item in ocr_info:
                item_count += 1
                if should_fix_to_answer(item, ocr_info):
                    item["pred_id_before_no_data_fix"] = item.get("pred_id")
                    item["pred_before_no_data_fix"] = item.get("pred")
                    item["pred_id"] = 3
                    item["pred"] = "ANSWER"
                    item["no_data_answer_fixed"] = True
                    fixed_count += 1

            fout.write(
                image_path
                + "\t"
                + json.dumps(data, ensure_ascii=False)
                + "\n"
            )
            image_count += 1

    summary = {
        "input": str(infer_path),
        "output": str(output_path),
        "image_count": image_count,
        "item_count": item_count,
        "fixed_count": fixed_count,
    }

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[DONE]")
    print(f"input={infer_path}")
    print(f"output={output_path}")
    print(f"summary={summary_path}")
    print(f"images={image_count} items={item_count} fixed={fixed_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    process_infer(args.infer, args.output)


if __name__ == "__main__":
    main()