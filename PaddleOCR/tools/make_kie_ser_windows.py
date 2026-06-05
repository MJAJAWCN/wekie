import argparse
import json
import math
from pathlib import Path

import cv2


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2 failed to read image: {path}")
    return img


def write_image(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"cv2 failed to write image: {path}")


def bbox_from_item(item):
    if "bbox" in item and len(item["bbox"]) == 4:
        return [int(round(v)) for v in item["bbox"]]
    pts = item.get("points", [])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def bbox_to_points(bbox):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def shift_item_to_window(item, crop_y):
    new_item = dict(item)
    bbox = bbox_from_item(item)
    local_bbox = [bbox[0], bbox[1] - crop_y, bbox[2], bbox[3] - crop_y]
    new_item["bbox"] = local_bbox
    new_item["points"] = bbox_to_points(local_bbox)

    new_item["logical_bbox"] = bbox
    new_item["window_crop_y"] = crop_y
    return new_item


def center_y(bbox):
    return (bbox[1] + bbox[3]) / 2.0


def height(bbox):
    return max(1, bbox[3] - bbox[1])


def estimate_tokens(text):
    text = text or ""
    if not text:
        return 0

    # Conservative estimate for LayoutXLM sentencepiece:
    # Chinese-like chars roughly count as one token; ASCII runs are cheaper but we keep it safe.
    tokens = 0
    ascii_run = 0

    for ch in text:
        if ord(ch) < 128 and ch.isalnum():
            ascii_run += 1
        else:
            if ascii_run:
                tokens += max(1, math.ceil(ascii_run / 4))
                ascii_run = 0
            if not ch.isspace():
                tokens += 1

    if ascii_run:
        tokens += max(1, math.ceil(ascii_run / 4))

    return max(1, tokens + 1)


def prepare_items(ocr_info):
    items = []
    for idx, item in enumerate(ocr_info):
        text = item.get("transcription", "")
        if not text:
            continue

        bbox = bbox_from_item(item)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        new_item = dict(item)
        new_item["_idx"] = idx
        new_item["_bbox"] = bbox
        new_item["_cy"] = center_y(bbox)
        new_item["_token_est"] = estimate_tokens(text)
        items.append(new_item)

    return sorted(items, key=lambda x: (x["_bbox"][1], x["_bbox"][0], x["_idx"]))


def build_rows(items):
    if not items:
        return []

    hs = sorted(height(x["_bbox"]) for x in items)
    median_h = hs[len(hs) // 2]
    row_tol = max(10, int(median_h * 0.75))

    rows = []
    for item in items:
        placed = False
        for row in rows:
            if abs(item["_cy"] - row["cy"]) <= row_tol:
                row["items"].append(item)
                ys = [x["_cy"] for x in row["items"]]
                row["cy"] = sum(ys) / len(ys)
                placed = True
                break

        if not placed:
            rows.append({"cy": item["_cy"], "items": [item]})

    for row_idx, row in enumerate(rows):
        row["items"].sort(key=lambda x: (x["_bbox"][0], x["_bbox"][1], x["_idx"]))
        bboxes = [x["_bbox"] for x in row["items"]]
        row["bbox"] = [
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        ]
        row["token_est"] = sum(x["_token_est"] for x in row["items"])
        row["row_idx"] = row_idx

    rows.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    for row_idx, row in enumerate(rows):
        row["row_idx"] = row_idx

    return rows


def window_height(rows, start, end):
    ys1 = min(rows[i]["bbox"][1] for i in range(start, end + 1))
    ys2 = max(rows[i]["bbox"][3] for i in range(start, end + 1))
    return ys2 - ys1


def build_windows(rows, max_tokens, max_window_height, overlap_rows):
    windows = []
    n = len(rows)
    start = 0

    while start < n:
        end = start
        token_sum = 0

        while end < n:
            next_tokens = token_sum + rows[end]["token_est"]
            next_height = window_height(rows, start, end)

            if end > start and next_tokens > max_tokens:
                break
            if end > start and next_height > max_window_height:
                break

            token_sum = next_tokens
            end += 1

        core_start = start
        core_end = max(start, end - 1)

        ctx_start = max(0, core_start - overlap_rows)
        ctx_end = min(n - 1, core_end + overlap_rows)

        windows.append({
            "core_start_row": core_start,
            "core_end_row": core_end,
            "context_start_row": ctx_start,
            "context_end_row": ctx_end,
        })

        start = core_end + 1

    return windows


def make_window_items(rows, win):
    selected = []
    seen = set()

    for row_i in range(win["context_start_row"], win["context_end_row"] + 1):
        for item in rows[row_i]["items"]:
            key = item.get("id", item["_idx"])
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)

    selected.sort(key=lambda x: (x["_bbox"][1], x["_bbox"][0], x["_idx"]))
    return selected


def strip_internal_fields(item):
    out = dict(item)
    for k in list(out.keys()):
        if k.startswith("_"):
            out.pop(k, None)
    return out

"""
作用：生成 window OCR 时，把值格里的 无资料 临时改成 回答：无资料，让 SER 更容易判成 ANSWER；
同时保留 original_transcription，后面结果里仍可还原原文。
"""

NO_DATA_VALUES = {
    "无资料",
    "暂无资料",
    "无数据",
    "暂无数据",
    "未提供",
    "未见资料",
}


def normalize_no_data_text(text):
    return "".join(str(text or "").split()).strip("。；;：:")


def is_no_data_text(text):
    return normalize_no_data_text(text) in NO_DATA_VALUES


def logic_box_says_value_cell(item):
    logic_box = item.get("logic_box")
    if not logic_box or len(logic_box) != 4:
        return False

    row_start, row_end, col_start, col_end = [int(x) for x in logic_box]

    # 全行跨列通常是 section 标题或段落正文，不在这里加“回答：”
    if col_start == 0 and col_end >= 3:
        return False

    # SDS 表格常见结构是 Q/A 或 Q/A/Q/A，值一般在第 1、3 列
    return col_start in (1, 3)


def enhance_no_data_for_ser(item):
    text = item.get("transcription", "")
    if not is_no_data_text(text):
        return item

    if not logic_box_says_value_cell(item):
        return item

    out = dict(item)
    out["original_transcription"] = text
    # out["transcription_for_ser_prompt"] = "回答：" + text
    # out["transcription"] = "回答：" + text
    out["transcription_for_ser_prompt"] = "这是一个回答"
    out["transcription"] = "这是一个回答"
    out["ser_prompt_enhanced"] = True
    out["ser_prompt_enhance_reason"] = "no_data_value_cell"
    return out

def process_one_group(group, output_root, max_tokens, max_window_height, overlap_rows, pad):
    image_path = Path(group["image"])
    ocr_path = Path(group["ocr_info"])

    logical_img = read_image(image_path)
    img_h, img_w = logical_img.shape[:2]

    ocr_info = read_json(ocr_path)
    items = prepare_items(ocr_info)
    rows = build_rows(items)
    windows = build_windows(rows, max_tokens, max_window_height, overlap_rows)

    out_img_dir = output_root / "images"
    out_ocr_dir = output_root / "ocr"
    out_map_dir = output_root / "maps"

    group_id = int(group["group_id"])
    produced = []

    for win_idx, win in enumerate(windows, start=1):
        win_items = make_window_items(rows, win)
        if not win_items:
            continue

        y1 = min(x["_bbox"][1] for x in win_items)
        y2 = max(x["_bbox"][3] for x in win_items)

        crop_y1 = max(0, y1 - pad)
        crop_y2 = min(img_h, y2 + pad)

        crop = logical_img[crop_y1:crop_y2, 0:img_w]

        stem = f"logical_table_{group_id:03d}_w{win_idx:03d}"
        img_out = out_img_dir / f"{stem}.jpg"
        ocr_out = out_ocr_dir / f"{stem}_ocr_info.json"
        map_out = out_map_dir / f"{stem}_map.json"

        local_items = []
        map_items = []

        core_row_min = win["core_start_row"]
        core_row_max = win["core_end_row"]

        for item in win_items:
            clean = strip_internal_fields(item)
            local = shift_item_to_window(clean, crop_y1)

            row_idx = None
            for r in rows:
                if any(x["_idx"] == item["_idx"] for x in r["items"]):
                    row_idx = r["row_idx"]
                    break

            local["window_id"] = stem
            local["is_core_item"] = bool(row_idx is not None and core_row_min <= row_idx <= core_row_max)
            
            # 这样改完后，SER 输入里看到的是 回答：无资料，但 item 上还保留：无资料
            # “无资料 -> ANSWER”和对应 key 保护放到合并后的结构后处理里做。
            # local = enhance_no_data_for_ser(local)

            local_items.append(local)

            map_items.append({
                "id": local.get("id"),
                "transcription": local.get("transcription"),
                "window_id": stem,
                "is_core_item": local["is_core_item"],
                "source_page_idx": local.get("source_page_idx"),
                "source_image_path": local.get("source_image_path"),
                "source_bbox": local.get("source_bbox"),
                "logical_bbox": local.get("logical_bbox"),
                "window_bbox": local.get("bbox"),
                "cell_id": local.get("cell_id"),
                "logic_box": local.get("logic_box"),
                "fragment_index": local.get("fragment_index"),
            })

        write_image(img_out, crop)
        write_json(ocr_out, local_items)
        write_json(map_out, {
            "window_id": stem,
            "group_id": group_id,
            "source_logical_image": str(image_path),
            "source_logical_ocr": str(ocr_path),
            "window_image": str(img_out),
            "window_ocr": str(ocr_out),
            "crop_x1": 0,
            "crop_y1": crop_y1,
            "crop_x2": img_w,
            "crop_y2": crop_y2,
            "core_start_row": win["core_start_row"],
            "core_end_row": win["core_end_row"],
            "context_start_row": win["context_start_row"],
            "context_end_row": win["context_end_row"],
            "item_count": len(local_items),
            "token_est": sum(x["_token_est"] for x in win_items),
            "items": map_items,
        })

        produced.append({
            "window_id": stem,
            "image": str(img_out),
            "ocr_info": str(ocr_out),
            "map": str(map_out),
            "item_count": len(local_items),
            "token_est": sum(x["_token_est"] for x in win_items),
            "crop_y1": crop_y1,
            "crop_y2": crop_y2,
            "core_start_row": win["core_start_row"],
            "core_end_row": win["core_end_row"],
            "context_start_row": win["context_start_row"],
            "context_end_row": win["context_end_row"],
        })

        print(
            f"[OK] {stem} rows={win['context_start_row']}-{win['context_end_row']} "
            f"core={win['core_start_row']}-{win['core_end_row']} "
            f"items={len(local_items)} tokens~{produced[-1]['token_est']} "
            f"crop_y={crop_y1}:{crop_y2}"
        )

    return produced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logical_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--max_tokens", type=int, default=430)
    parser.add_argument("--max_window_height", type=int, default=1600)
    parser.add_argument("--overlap_rows", type=int, default=2)
    parser.add_argument("--pad", type=int, default=32)
    args = parser.parse_args()

    logical_root = Path(args.logical_root)
    output_root = Path(args.output_root)

    manifest = read_json(logical_root / "manifest.json")

    all_windows = []
    for group in manifest["groups"]:
        produced = process_one_group(
            group=group,
            output_root=output_root,
            max_tokens=args.max_tokens,
            max_window_height=args.max_window_height,
            overlap_rows=args.overlap_rows,
            pad=args.pad,
        )
        all_windows.extend(produced)

    out_manifest = {
        "source_logical_root": str(logical_root),
        "max_tokens": args.max_tokens,
        "max_window_height": args.max_window_height,
        "overlap_rows": args.overlap_rows,
        "pad": args.pad,
        "window_count": len(all_windows),
        "windows": all_windows,
    }

    write_json(output_root / "manifest.json", out_manifest)

    print(f"[DONE] output_root={output_root}")
    print(f"       image_dir={output_root / 'images'}")
    print(f"       ocr_info_dir={output_root / 'ocr'}")
    print(f"       map_dir={output_root / 'maps'}")
    print(f"       windows={len(all_windows)}")


if __name__ == "__main__":
    main()