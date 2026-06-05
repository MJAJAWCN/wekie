import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def bbox_to_points(bbox):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def union_bbox(bboxes):
    return [
        int(min(b[0] for b in bboxes)),
        int(min(b[1] for b in bboxes)),
        int(max(b[2] for b in bboxes)),
        int(max(b[3] for b in bboxes)),
    ]


def shift_bbox(bbox, dx, dy):
    return [
        int(round(bbox[0] + dx)),
        int(round(bbox[1] + dy)),
        int(round(bbox[2] + dx)),
        int(round(bbox[3] + dy)),
    ]


def normalize_text(text):
    return (text or "").strip()


def bbox_height(b):
    return max(1, int(b[3] - b[1]))


def bbox_width(b):
    return max(1, int(b[2] - b[0]))


def center_y(b):
    return (b[1] + b[3]) / 2.0


def x_overlap_ratio(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return overlap / float(max(1, min(bbox_width(a), bbox_width(b))))


def has_colon(text):
    return any(ch in (text or "") for ch in ["：", ":", "﹕", "︰"])


def has_checkbox_or_option(text):
    markers = ["□", "☑", "☒", "√", "✓", "无异常", "异常", "是否", "是", "否"]
    return any(m in (text or "") for m in markers)


def looks_like_section_title(text):
    text = normalize_text(text)
    return bool(re.match(r"^([一二三四五六七八九十]+|\d+)[、.．]\s*\S{1,40}$", text))


def looks_like_table_value(text):
    text = normalize_text(text)
    if not text:
        return True
    patterns = [
        r"^\d{1,2}[:：]\d{2}$",
        r"^\d{1,4}([./-]\d{1,2}){1,2}$",
        r"^\d+(\.\d+)?\s*(℃|°C|%|%RH|RH)?$",
        r"^[\dA-Za-z_.:/：-]{1,10}$",
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


def sort_cell_items_for_merge(ocr_items):
    valid = []
    for idx, item in enumerate(ocr_items or []):
        text = normalize_text(item.get("text", ""))
        bbox = item.get("bbox", [])
        if not text or not bbox or len(bbox) != 4:
            continue
        copied = dict(item)
        copied["_source_order"] = idx
        copied["bbox"] = [int(round(v)) for v in bbox]
        valid.append(copied)

    return sorted(valid, key=lambda x: (x["bbox"][1], x["bbox"][0], x["_source_order"]))


def make_merged_item(items, separator=""):
    texts = [normalize_text(i.get("text", "")) for i in items if normalize_text(i.get("text", ""))]
    bboxes = [i["bbox"] for i in items]
    first = dict(items[0])
    first["text"] = separator.join(texts)
    first["bbox"] = union_bbox(bboxes)
    first["score"] = min(float(i.get("score", 1.0)) for i in items)
    first["ocr_id"] = "+".join(str(i.get("ocr_id", "")) for i in items if i.get("ocr_id"))
    first["source_ocr_ids"] = [i.get("ocr_id", "") for i in items]
    first["merged_from_count"] = len(items)
    return first


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

    return sorted(result, key=lambda x: x.get("_source_order", 0))


"""
同一个 MinerU cell 内
如果两个 OCR item 横向紧挨着
且其中一个是公式/单位
就合并成一个 OCR item

例如把：

黏度
<eq>(mm²/s)</eq>
合成：

黏度 <eq>(mm²/s)</eq>
这样后面 SER 会把它当成一个完整字段 key。
"""


def is_formula_or_unit_text(text):
    text = normalize_text(text)
    if not text:
        return False

    if "<eq>" in text or "</eq>" in text:
        return True

    unit_markers = [
        "℃", "KPa", "kPa", "MPa", "mJ", "KJ", "kJ",
        "mol", "mm", "cm", "m/s", "mg", "kg", "%", "pH",
    ]
    if any(x in text for x in unit_markers):
        return True

    # 括号里的短单位，例如 (空气=1)、(mm2/s)、（%）
    stripped = text.strip()
    if (
        len(stripped) <= 30
        and (
            (stripped.startswith("(") and stripped.endswith(")"))
            or (stripped.startswith("（") and stripped.endswith("）"))
        )
    ):
        return True

    return False


def y_overlap_ratio_for_merge(a, b):
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]
    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    denom = max(1, min(ay2 - ay1, by2 - by1))
    return overlap / denom


def can_merge_inline_in_cell(left, right):
    left_box = left.get("bbox", [])
    right_box = right.get("bbox", [])
    if len(left_box) != 4 or len(right_box) != 4:
        return False

    left_text = normalize_text(left.get("text", ""))
    right_text = normalize_text(right.get("text", ""))
    if not left_text or not right_text:
        return False

    # 只处理同一视觉行里的横向紧邻，不处理上下续行
    if y_overlap_ratio_for_merge(left_box, right_box) < 0.55:
        return False

    gap = right_box[0] - left_box[2]
    avg_h = ((left_box[3] - left_box[1]) + (right_box[3] - right_box[1])) / 2.0

    # 黏度和单位这种通常 gap 极小；放宽到 0.35 * 字高，但最多 18px
    if gap < -3:
        return False
    if gap > min(18, max(3, 0.35 * avg_h)):
        return False

    # 保守：必须有一边是公式/单位
    if not (is_formula_or_unit_text(left_text) or is_formula_or_unit_text(right_text)):
        return False

    # 避免把长说明文字横向拼乱
    if len(left_text) + len(right_text) > 80:
        return False

    return True


def merge_inline_items_in_cell(ocr_items, separator=" "):
    items = sort_cell_items_for_merge(ocr_items)
    if len(items) <= 1:
        return items

    merged = []
    i = 0
    while i < len(items):
        cur = items[i]
        chain = [cur]
        j = i + 1

        while j < len(items):
            nxt = items[j]
            if not can_merge_inline_in_cell(chain[-1], nxt):
                break
            chain.append(nxt)
            j += 1

        if len(chain) > 1:
            merged.append(make_merged_item(chain, separator=separator))
        else:
            merged.append(cur)

        i = j

    return sorted(merged, key=lambda x: x.get("_source_order", 0))


def split_key_value_item(text, bbox, score=1.0, extra=None):
    extra = extra or {}
    text = normalize_text(text)
    if not text:
        return []

    split_pos = -1
    for ch in ["：", ":", "﹕", "︰"]:
        pos = text.find(ch)
        if pos > 0:
            split_pos = pos
            break

    def one_piece(piece_text, piece_bbox, suffix):
        item_id = f"{extra['base_id']}_{suffix}"
        return {
            **extra,
            "id": item_id,
            "ocr_id": item_id,
            "source_ocr_id": extra.get("source_ocr_id", ""),
            "transcription": piece_text,
            "bbox": [int(v) for v in piece_bbox],
            "points": bbox_to_points(piece_bbox),
            "score": float(score),
        }

    if split_pos <= 0 or split_pos >= len(text) - 1:
        return [one_piece(text, bbox, "p0")]

    left_char = text[split_pos - 1]
    right_char = text[split_pos + 1]
    if left_char.isdigit() and right_char.isdigit():
        return [one_piece(text, bbox, "p0")]

    key_text = text[:split_pos + 1].strip()
    value_text = text[split_pos + 1:].strip()
    if not key_text or not value_text:
        return [one_piece(text, bbox, "p0")]

    if len(text) > 40:
        return [one_piece(text, bbox, "p0")]

    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(1, x2 - x1)

    key_ratio = len(key_text) / float(max(1, len(key_text) + len(value_text)))
    split_x = int(x1 + width * key_ratio)
    split_x = max(x1 + 1, min(split_x, x2 - 1))

    return [
        one_piece(key_text, [x1, y1, split_x, y2], "key"),
        one_piece(value_text, [split_x, y1, x2, y2], "value"),
    ]


def find_one(auto_dir, suffix):
    files = sorted(Path(auto_dir).glob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"Cannot find *{suffix} in {auto_dir}")
    return files[0]


def extract_image_path_from_block(block):
    hits = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("image_path", "img_path") and v:
                    hits.append(v)
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                if isinstance(v, (dict, list)):
                    walk(v)

    walk(block)
    return hits[0] if hits else None


def extract_middle_fragments(middle_json):
    data = read_json(middle_json)
    fragments = []
    seen = set()

    for page_i, page in enumerate(data.get("pdf_info", [])):
        page_idx = page.get("page_idx", page_i)

        # Prefer preproc_blocks, because it consistently carries table image paths in 0602 output.
        for sec in ["preproc_blocks", "para_blocks"]:
            for block_i, block in enumerate(page.get(sec, []) or []):
                if block.get("type") != "table":
                    continue

                bbox = tuple(block.get("bbox") or [])
                image_path = extract_image_path_from_block(block)
                key = (page_idx, bbox, image_path)

                if key in seen:
                    continue
                seen.add(key)

                fragments.append({
                    "page_idx": page_idx,
                    "section": sec,
                    "block_index": block_i,
                    "bbox": list(bbox),
                    "image_path": image_path,
                })

    fragments = [f for f in fragments if f.get("image_path")]
    fragments.sort(key=lambda f: (f["page_idx"], f["bbox"][1], f["bbox"][0]))
    return fragments


def extract_wired_tables(model_json):
    data = read_json(model_json)
    tables = []

    if not isinstance(data, list):
        raise ValueError("Expected latest MinerU model.json top-level to be a list.")

    for page_idx, page in enumerate(data):
        for det_idx, det in enumerate(page.get("layout_dets", []) or []):
            cells = det.get("wired_cell_ocr_items")
            if not cells:
                continue
            tables.append({
                "page_idx": page_idx,
                "det_idx": det_idx,
                "table_id": det.get("index", det_idx),
                "table_bbox": det.get("bbox", []),
                "selected_model": det.get("selected_model", ""),
                "cells": cells,
            })

    tables.sort(key=lambda t: (
        t["page_idx"],
        t["table_bbox"][1] if len(t["table_bbox"]) == 4 else 0,
        t["table_bbox"][0] if len(t["table_bbox"]) == 4 else 0,
    ))
    return tables


def extract_content_groups(content_json):
    data = read_json(content_json)
    tables = [x for x in data if x.get("type") == "table"]

    groups = []
    cur = None

    for idx, t in enumerate(tables):
        img_path = t.get("img_path") or ""
        starts_new = bool(img_path) or cur is None

        if starts_new:
            if cur is not None:
                groups.append(cur)
            cur = {
                "content_start_index": idx,
                "caption": t.get("table_caption", ""),
                "content_items": [],
            }

        cur["content_items"].append({
            "content_index": idx,
            "page_idx": t.get("page_idx"),
            "bbox": t.get("bbox", []),
            "img_path": img_path,
            "body_len": len(t.get("table_body") or ""),
        })

    if cur is not None:
        groups.append(cur)

    return groups


def resolve_image_path(images_dir, image_path):
    p = Path(image_path)
    if p.is_absolute() and p.exists():
        return p

    name = p.name
    candidates = [
        Path(images_dir) / name,
        Path(images_dir).parent / image_path,
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Cannot resolve table image: {image_path}")


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


def build_virtual_image(fragments, images_dir, output_image_path, gap):
    loaded = []
    max_w = 0
    total_h = 0

    for frag in fragments:
        img_path = resolve_image_path(images_dir, frag["image_path"])
        img = read_image(img_path)
        h, w = img.shape[:2]
        loaded.append((img_path, img, w, h))
        max_w = max(max_w, w)
        total_h += h

    total_h += gap * max(0, len(loaded) - 1)
    canvas = np.full((total_h, max_w, 3), 255, dtype=np.uint8)

    y = 0
    placed = []
    for frag, (img_path, img, w, h) in zip(fragments, loaded):
        x = 0
        canvas[y:y + h, x:x + w] = img
        placed.append({
            **frag,
            "source_image_path": str(img_path),
            "virtual_x_offset": x,
            "virtual_y_offset": y,
            "image_width": w,
            "image_height": h,
        })
        y += h + gap

    write_image(output_image_path, canvas)
    return placed, max_w, total_h


def flatten_table_items(group_idx, fragment_idx, table, placed_fragment, merge_multiline=True):
    out = []
    mapping = []

    dx = placed_fragment["virtual_x_offset"]
    dy = placed_fragment["virtual_y_offset"]

    for cell in table["cells"]:
        cell_id = cell.get("cell_id", "")
        source_cell_bbox = cell.get("cell_bbox", [])
        virtual_cell_bbox = shift_bbox(source_cell_bbox, dx, dy) if len(source_cell_bbox) == 4 else []

        ocr_items = cell.get("ocr_items", []) or []
        if merge_multiline:
            # 合并跨行
            ocr_items = merge_multiline_items_in_cell(ocr_items)
            # 合并黏度（单位）这种横向紧邻的
            ocr_items = merge_inline_items_in_cell(ocr_items)

        for local_i, ocr in enumerate(ocr_items):
            text = normalize_text(ocr.get("text", ""))
            source_bbox = ocr.get("bbox", [])
            if not text or not source_bbox or len(source_bbox) != 4:
                continue

            source_bbox = [int(round(v)) for v in source_bbox]
            virtual_bbox = shift_bbox(source_bbox, dx, dy)

            source_ocr_id = ocr.get("ocr_id", f"{cell_id}_o{local_i}")
            base_id = f"g{group_idx:03d}_f{fragment_idx:03d}_{source_ocr_id}".replace("+", "_")

            extra = {
                "base_id": base_id,
                "group_id": group_idx,
                "fragment_index": fragment_idx,
                "source_page_idx": placed_fragment["page_idx"],
                "source_image_path": placed_fragment["source_image_path"],
                "source_fragment_bbox": placed_fragment.get("bbox", []),
                "source_bbox": source_bbox,
                "virtual_bbox": virtual_bbox,
                "source_ocr_id": source_ocr_id,
                "source_ocr_ids": ocr.get("source_ocr_ids", [source_ocr_id]),
                "merged_from_count": ocr.get("merged_from_count", 1),
                "cell_id": cell_id,
                "source_cell_bbox": source_cell_bbox,
                "cell_bbox": virtual_cell_bbox,
                "logic_box": cell.get("logic_box", []),
                "table_id": table.get("table_id"),
                "table_det_idx": table.get("det_idx"),
                "table_bbox": table.get("table_bbox", []),
            }

            pieces = split_key_value_item(
                text=text,
                bbox=virtual_bbox,
                score=ocr.get("score", 1.0),
                extra=extra,
            )

            for p in pieces:
                p.pop("base_id", None)
                out.append(p)
                mapping.append({
                    "id": p["id"],
                    "transcription": p["transcription"],
                    "group_id": group_idx,
                    "fragment_index": fragment_idx,
                    "source_page_idx": placed_fragment["page_idx"],
                    "source_image_path": placed_fragment["source_image_path"],
                    "source_bbox": source_bbox,
                    "virtual_bbox": p["bbox"],
                    "cell_id": cell_id,
                    "logic_box": cell.get("logic_box", []),
                    "source_ocr_id": source_ocr_id,
                })

    return out, mapping


def prepare(auto_dir, output_root, gap=24, merge_multiline=True):
    auto_dir = Path(auto_dir)
    output_root = Path(output_root)

    model_json = find_one(auto_dir, "_model.json")
    middle_json = find_one(auto_dir, "_middle.json")
    content_json = find_one(auto_dir, "_content_list.json")
    images_dir = auto_dir / "images"

    out_img_dir = output_root / "images"
    out_ocr_dir = output_root / "ocr"
    out_map_dir = output_root / "maps"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ocr_dir.mkdir(parents=True, exist_ok=True)
    out_map_dir.mkdir(parents=True, exist_ok=True)

    groups = extract_content_groups(content_json)
    middle_fragments = extract_middle_fragments(middle_json)
    wired_tables = extract_wired_tables(model_json)

    content_fragment_count = sum(len(g["content_items"]) for g in groups)
    if content_fragment_count != len(middle_fragments):
        print(f"[WARN] content table count={content_fragment_count}, middle fragments with image={len(middle_fragments)}")
    if content_fragment_count != len(wired_tables):
        print(f"[WARN] content table count={content_fragment_count}, wired tables={len(wired_tables)}")

    manifest = {
        "auto_dir": str(auto_dir),
        "model_json": str(model_json),
        "middle_json": str(middle_json),
        "content_json": str(content_json),
        "gap": gap,
        "merge_multiline": merge_multiline,
        "groups": [],
    }

    frag_cursor = 0
    table_cursor = 0

    for group_idx, group in enumerate(groups, start=1):
        n = len(group["content_items"])
        group_fragments = middle_fragments[frag_cursor:frag_cursor + n]
        group_tables = wired_tables[table_cursor:table_cursor + n]

        if len(group_fragments) != n or len(group_tables) != n:
            raise RuntimeError(
                f"group {group_idx}: expected {n} fragments/tables, "
                f"got fragments={len(group_fragments)}, tables={len(group_tables)}"
            )

        image_name = f"logical_table_{group_idx:03d}.jpg"
        ocr_name = f"logical_table_{group_idx:03d}_ocr_info.json"
        map_name = f"logical_table_{group_idx:03d}_map.json"

        placed, virtual_w, virtual_h = build_virtual_image(
            group_fragments,
            images_dir,
            out_img_dir / image_name,
            gap=gap,
        )

        kie_items = []
        item_map = []

        for frag_i, (table, placed_fragment) in enumerate(zip(group_tables, placed), start=1):
            items, maps = flatten_table_items(
                group_idx=group_idx,
                fragment_idx=frag_i,
                table=table,
                placed_fragment=placed_fragment,
                merge_multiline=merge_multiline,
            )
            kie_items.extend(items)
            item_map.extend(maps)

        write_json(out_ocr_dir / ocr_name, kie_items)
        write_json(out_map_dir / map_name, {
            "group_id": group_idx,
            "image": str(out_img_dir / image_name),
            "ocr_info": str(out_ocr_dir / ocr_name),
            "virtual_width": virtual_w,
            "virtual_height": virtual_h,
            "content_items": group["content_items"],
            "fragments": placed,
            "items": item_map,
        })

        manifest["groups"].append({
            "group_id": group_idx,
            "image": str(out_img_dir / image_name),
            "ocr_info": str(out_ocr_dir / ocr_name),
            "map": str(out_map_dir / map_name),
            "fragment_count": n,
            "item_count": len(kie_items),
            "virtual_width": virtual_w,
            "virtual_height": virtual_h,
        })

        print(
            f"[OK] group={group_idx:03d} fragments={n} "
            f"items={len(kie_items)} image={out_img_dir / image_name}"
        )

        frag_cursor += n
        table_cursor += n

    write_json(output_root / "manifest.json", manifest)
    print(f"[DONE] output_root={output_root}")
    print(f"       image_dir={out_img_dir}")
    print(f"       ocr_info_dir={out_ocr_dir}")
    print(f"       map_dir={out_map_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto_dir", required=True, help="Latest MinerU auto output dir, e.g. ...\\sds_first10_pages\\auto")
    parser.add_argument("--output_root", required=True, help="Output root for KIE logical table inputs")
    parser.add_argument("--gap", type=int, default=24)
    parser.add_argument("--no_merge_multiline", action="store_true")
    args = parser.parse_args()

    prepare(
        auto_dir=args.auto_dir,
        output_root=args.output_root,
        gap=args.gap,
        merge_multiline=not args.no_merge_multiline,
    )


if __name__ == "__main__":
    main()