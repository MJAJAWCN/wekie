"""
读取 infer_fixed_no_data_answer_rule.txt
-> 按 item id 合并多个 window 的预测
-> 对每个 vote 判断上下文是否完整
-> 如果有 “非 O 且上下左右上下文可信” 的 vote，优先采用它
-> 否则回退到旧规则：core 优先 + 多数投票
-> 输出每张逻辑大表 merged SER JSON
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


PRED_ID = {
    "O": 0,
    "OTHER": 0,
    "QUESTION": 1,
    "ANSWER": 3,
    "HEADER": 5,
}

TIE_PRIORITY = {
    "QUESTION": 4,
    "ANSWER": 3,
    "HEADER": 2,
    "O": 1,
    "OTHER": 1,
}


def read_infer(path):
    records = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                print(f"[WARN] line {line_no}: no tab separator, skipped")
                continue

            image_path, payload = line.split("\t", 1)
            data = json.loads(payload)
            ocr_info = data.get("ocr_info", [])

            records.append({
                "line_no": line_no,
                "image_path": image_path,
                "ocr_info": ocr_info,
            })

    return records


def normalize_pred(pred):
    pred = pred or "O"
    pred = str(pred).upper()
    if pred == "OTHER":
        return "O"
    return pred


def get_group_id(item):
    if item.get("group_id") is not None:
        return int(item["group_id"])

    window_id = item.get("window_id", "")
    parts = window_id.split("_")
    for i, p in enumerate(parts):
        if p == "table" and i + 1 < len(parts):
            return int(parts[i + 1])

    raise RuntimeError(f"cannot infer group_id for item id={item.get('id')}")


def get_bbox(item):
    bbox = item.get("bbox")
    if bbox and len(bbox) == 4:
        return [int(round(x)) for x in bbox]
    return None


def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def estimate_window_size(items):
    max_x = 0
    max_y = 0
    for item in items:
        bbox = get_bbox(item)
        if not bbox:
            continue
        max_x = max(max_x, bbox[2])
        max_y = max(max_y, bbox[3])
    return max_x, max_y


def has_vertical_context(item, window_items, margin=80):
    bbox = get_bbox(item)
    if not bbox:
        return False

    _, cy = bbox_center(bbox)
    _, win_h = estimate_window_size(window_items)

    has_above = False
    has_below = False

    for other in window_items:
        if other is item:
            continue
        obox = get_bbox(other)
        if not obox:
            continue
        _, ocy = bbox_center(obox)
        if ocy < cy:
            has_above = True
        if ocy > cy:
            has_below = True

    margin_ok = bbox[1] > margin and (win_h <= 0 or bbox[3] < win_h - margin)
    return (has_above and has_below) or margin_ok


def has_horizontal_context_relaxed(item, window_items, margin=80):
    bbox = get_bbox(item)
    if not bbox:
        return False

    cx, _ = bbox_center(bbox)
    win_w, _ = estimate_window_size(window_items)

    has_left = False
    has_right = False

    for other in window_items:
        if other is item:
            continue
        obox = get_bbox(other)
        if not obox:
            continue
        ocx, _ = bbox_center(obox)
        if ocx < cx:
            has_left = True
        if ocx > cx:
            has_right = True

    logic_box = item.get("logic_box")
    col_start = None
    if logic_box and len(logic_box) == 4:
        col_start = int(logic_box[2])

    if col_start == 0:
        margin_ok = win_w <= 0 or bbox[2] < win_w - margin
        return has_right or margin_ok

    margin_ok = bbox[0] > margin and (win_w <= 0 or bbox[2] < win_w - margin)
    return (has_left and has_right) or margin_ok


def has_full_context(item, window_items):
    return (
        has_vertical_context(item, window_items)
        and has_horizontal_context_relaxed(item, window_items)
    )


def make_vote(item, image_path, window_items):
    pred = normalize_pred(item.get("pred"))
    return {
        "window_id": item.get("window_id"),
        "image_path": image_path,
        "pred": pred,
        "pred_id": int(item.get("pred_id", PRED_ID.get(pred, 0))),
        "is_core_item": bool(item.get("is_core_item", False)),
        "has_full_context": bool(has_full_context(item, window_items)),
        "has_vertical_context": bool(has_vertical_context(item, window_items)),
        "has_horizontal_context": bool(has_horizontal_context_relaxed(item, window_items)),
        "no_data_answer_fixed": bool(item.get("no_data_answer_fixed", False)),
        "pred_before_no_data_fix": item.get("pred_before_no_data_fix"),
        "pred_id_before_no_data_fix": item.get("pred_id_before_no_data_fix"),
        "bbox": item.get("bbox"),
    }


def choose_by_counts(votes):
    counts = Counter(normalize_pred(v.get("pred")) for v in votes)

    def sort_key(pred):
        return (
            counts[pred],
            TIE_PRIORITY.get(pred, 0),
        )

    final_pred = max(counts.keys(), key=sort_key)
    return final_pred, PRED_ID.get(final_pred, 0), counts


def vote_final(votes):
    context_non_o_votes = [
        v for v in votes
        if normalize_pred(v.get("pred")) != "O" and v.get("has_full_context")
    ]

    if context_non_o_votes:
        final_pred, final_pred_id, counts = choose_by_counts(context_non_o_votes)
        return final_pred, final_pred_id, context_non_o_votes, counts, "context_non_o"

    core_votes = [v for v in votes if v.get("is_core_item")]
    usable_votes = core_votes if core_votes else votes
    final_pred, final_pred_id, counts = choose_by_counts(usable_votes)
    return final_pred, final_pred_id, usable_votes, counts, "core_or_all"


def base_item_from_votes(items):
    core_items = [x for x in items if x.get("is_core_item")]
    chosen = core_items[0] if core_items else items[0]

    keep_keys = [
        "id",
        "ocr_id",
        "transcription",
        "group_id",
        "fragment_index",
        "source_page_idx",
        "source_image_path",
        "source_fragment_bbox",
        "source_bbox",
        "virtual_bbox",
        "logical_bbox",
        "cell_id",
        "source_cell_bbox",
        "cell_bbox",
        "logic_box",
        "table_id",
        "table_det_idx",
        "table_bbox",
        "score",
    ]

    out = {}
    for key in keep_keys:
        if key in chosen:
            out[key] = chosen[key]

    return out


def merge_records(records):
    grouped_items = defaultdict(list)
    window_items_by_image = {}

    for record in records:
        image_path = record["image_path"]
        ocr_info = record["ocr_info"]
        window_items_by_image[image_path] = ocr_info

        for item in ocr_info:
            item_id = item.get("id") or item.get("ocr_id")
            if not item_id:
                print("[WARN] item without id/ocr_id skipped")
                continue
            copied = dict(item)
            copied["_image_path"] = image_path
            grouped_items[item_id].append(copied)

    merged_by_group = defaultdict(list)
    conflict_items = []
    duplicate_count = 0
    changed_by_merge = 0
    no_data_fixed_final = 0
    context_non_o_used = 0

    for item_id, items in grouped_items.items():
        votes = []
        for item in items:
            image_path = item["_image_path"]
            window_items = window_items_by_image.get(image_path, [])
            votes.append(make_vote(item, image_path, window_items))

        final_pred, final_pred_id, usable_votes, counts, merge_reason = vote_final(votes)

        if merge_reason == "context_non_o":
            context_non_o_used += 1

        if len(items) > 1:
            duplicate_count += 1

        raw_preds = {normalize_pred(v.get("pred")) for v in votes}
        if len(raw_preds) > 1:
            conflict_items.append({
                "id": item_id,
                "transcription": items[0].get("transcription"),
                "final_pred": final_pred,
                "merge_reason": merge_reason,
                "votes": votes,
                "usable_vote_counts": dict(counts),
            })

        chosen_pred = normalize_pred(items[0].get("pred"))
        if final_pred != chosen_pred:
            changed_by_merge += 1

        if final_pred == "ANSWER" and any(v.get("no_data_answer_fixed") for v in votes):
            no_data_fixed_final += 1

        out = base_item_from_votes(items)
        out["final_pred"] = final_pred
        out["final_pred_id"] = final_pred_id
        out["window_votes"] = votes
        out["merge_reason"] = merge_reason
        out["merge_used_core_votes"] = any(v.get("is_core_item") for v in usable_votes)
        out["merge_vote_counts"] = dict(counts)
        out["merge_duplicate_count"] = len(items)

        group_id = get_group_id(out)
        merged_by_group[group_id].append(out)

    stats = {
        "unique_item_count": len(grouped_items),
        "duplicate_item_count": duplicate_count,
        "changed_by_merge_count": changed_by_merge,
        "conflict_item_count": len(conflict_items),
        "no_data_fixed_final_count": no_data_fixed_final,
        "context_non_o_used_count": context_non_o_used,
        "conflict_items_sample": conflict_items[:80],
    }

    return merged_by_group, stats


def sort_items(items):
    def key(item):
        bbox = item.get("logical_bbox") or item.get("virtual_bbox") or item.get("bbox") or [0, 0, 0, 0]
        return (
            int(item.get("fragment_index", 0) or 0),
            int(bbox[1]),
            int(bbox[0]),
            str(item.get("id", "")),
        )

    return sorted(items, key=key)


def write_outputs(merged_by_group, stats, output_root, infer_path):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_infer": str(Path(infer_path)),
        "merge_strategy": "context_non_o_first_then_core_or_all",
        "groups": [],
        "global_stats": stats,
    }

    for group_id in sorted(merged_by_group):
        items = sort_items(merged_by_group[group_id])
        pred_counts = Counter(item.get("final_pred", "O") for item in items)
        reason_counts = Counter(item.get("merge_reason", "") for item in items)

        out = {
            "group_id": group_id,
            "source_infer": str(Path(infer_path)),
            "item_count": len(items),
            "final_pred_counts": dict(pred_counts),
            "merge_reason_counts": dict(reason_counts),
            "items": items,
        }

        out_name = f"logical_table_{group_id:03d}_ser_merged.json"
        out_path = output_root / out_name
        out_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest["groups"].append({
            "group_id": group_id,
            "output": str(out_path),
            "item_count": len(items),
            "final_pred_counts": dict(pred_counts),
            "merge_reason_counts": dict(reason_counts),
        })

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    records = read_infer(args.infer)
    merged_by_group, stats = merge_records(records)
    manifest_path = write_outputs(
        merged_by_group=merged_by_group,
        stats=stats,
        output_root=args.output_root,
        infer_path=args.infer,
    )

    print("[DONE]")
    print(f"infer={args.infer}")
    print(f"output_root={args.output_root}")
    print(f"manifest={manifest_path}")
    print(f"unique_items={stats['unique_item_count']}")
    print(f"duplicate_items={stats['duplicate_item_count']}")
    print(f"conflict_items={stats['conflict_item_count']}")
    print(f"changed_by_merge={stats['changed_by_merge_count']}")
    print(f"no_data_fixed_final={stats['no_data_fixed_final_count']}")
    print(f"context_non_o_used={stats['context_non_o_used_count']}")

    for group_id in sorted(merged_by_group):
        counts = Counter(item.get("final_pred", "O") for item in merged_by_group[group_id])
        reasons = Counter(item.get("merge_reason", "") for item in merged_by_group[group_id])
        print(f"group={group_id:03d} items={len(merged_by_group[group_id])} counts={dict(counts)} reasons={dict(reasons)}")


if __name__ == "__main__":
    main()