"""
作用：

读取规则 RE 结果：
kie_re_rule_23_main_only
读取 RE 模型输出：
infer.txt
通过 item 的 ocr_id/id/transcription/bbox 尝试映射模型关系到 merged item
剔除子表 item
分成：
accepted_rule_relations：规则关系，默认保留
model_only_relations：模型补充出来、规则没有的
conflict_relations：同一个 answer，模型和规则给了不同 question
final_relations：规则关系 + 无冲突模型补充关系
still_unlinked_answers：规则和模型都没连上的 answer
"""

import argparse
import json
from pathlib import Path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def text_of(item):
    return item.get("transcription") or item.get("text") or ""


def bbox_of(item):
    b = item.get("logical_bbox") or item.get("virtual_bbox") or item.get("bbox") or item.get("points")
    if isinstance(b, list) and len(b) == 4:
        return b
    return None


def norm_box(box):
    if not box:
        return None
    return tuple(int(round(float(x))) for x in box)


def item_key(item):
    return (
        text_of(item),
        norm_box(bbox_of(item)),
    )


def load_model_re_infer(infer_path):
    rows = []
    with open(infer_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            image_path, payload = line.split("\t", 1)
            data = json.loads(payload)
            rows.append({
                "image_path": image_path,
                "ocr_info": data.get("ocr_info") or data,
            })
    return rows


def iter_model_pairs(model_row):
    ocr_info = model_row.get("ocr_info") or []

    for pair in ocr_info:
        if isinstance(pair, list) and len(pair) == 2:
            yield pair[0], pair[1]
        elif isinstance(pair, tuple) and len(pair) == 2:
            yield pair[0], pair[1]


def build_merged_lookup(items):
    by_key = {}
    by_text = {}

    for item in items:
        if item.get("in_subtable"):
            continue

        key = item_key(item)
        by_key.setdefault(key, []).append(item)

        text = text_of(item)
        if text:
            by_text.setdefault(text, []).append(item)

    return by_key, by_text


def resolve_model_item(model_item, by_key, by_text):
    key = item_key(model_item)
    candidates = by_key.get(key) or []
    if len(candidates) == 1:
        return candidates[0]

    text = text_of(model_item)
    candidates = by_text.get(text) or []
    if len(candidates) == 1:
        return candidates[0]

    return None


"""
判断模型预测的 q/a 是否真的在附近。
只有同 cell、左邻、上邻、紧邻 section 才允许模型补入。
远距离模型关系会被拒绝。
"""

def logic_box(item):
    lb = item.get("logic_box")
    if isinstance(lb, list) and len(lb) == 4:
        return lb
    return None


def interval_overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def covers(a0, a1, b0, b1):
    return a0 <= b0 and a1 >= b1


def is_split_key_value_pair(q, a):
    qid = q.get("id") or ""
    aid = a.get("id") or ""
    if not qid.endswith("_key") or not aid.endswith("_value"):
        return False
    return qid[:-4] == aid[:-6]


def is_same_cell_near(q, a):
    if q.get("cell_id") != a.get("cell_id"):
        return False

    if is_split_key_value_pair(q, a):
        return True

    qb = bbox_of(q)
    ab = bbox_of(a)
    if not qb or not ab:
        return False

    q_left, q_top, q_right, q_bottom = qb
    a_left, a_top, a_right, a_bottom = ab

    y_overlap = interval_overlap(q_top, q_bottom, a_top, a_bottom)
    x_overlap = interval_overlap(q_left, q_right, a_left, a_right)

    q_is_left = q_right <= a_left + 3 and y_overlap > 0
    q_is_above = q_bottom <= a_top + 3 and x_overlap > 0

    return q_is_left or q_is_above


def is_left_neighbor(q, a):
    qlb = logic_box(q)
    alb = logic_box(a)
    if not qlb or not alb:
        return False

    qr0, qr1, qc0, qc1 = qlb
    ar0, ar1, ac0, ac1 = alb

    direct_col = qc1 + 1 == ac0
    if not direct_col:
        return False

    row_overlap = interval_overlap(qr0, qr1, ar0, ar1)
    if row_overlap <= 0:
        return False

    return True


def is_above_neighbor(q, a):
    qlb = logic_box(q)
    alb = logic_box(a)
    if not qlb or not alb:
        return False

    qr0, qr1, qc0, qc1 = qlb
    ar0, ar1, ac0, ac1 = alb

    direct_row = qr1 + 1 == ar0
    if not direct_row:
        return False

    col_overlap = interval_overlap(qc0, qc1, ac0, ac1)
    if col_overlap <= 0:
        return False

    return True


def is_section_like_question(q):
    text = text_of(q).strip()
    if not text:
        return False

    prefixes = (
        "一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、",
        "十一、", "十二、", "十三、", "十四、", "十五、",
        "1、", "2、", "3、", "4、", "5、", "6、", "7、", "8、", "9、",
    )
    return text.startswith(prefixes)


def is_near_section_relation(q, a):
    if not is_section_like_question(q):
        return False

    qlb = logic_box(q)
    alb = logic_box(a)
    if not qlb or not alb:
        return False

    qr0, qr1, qc0, qc1 = qlb
    ar0, ar1, ac0, ac1 = alb

    # section 只能管紧接着的下一行，不能跨远。
    if ar0 != qr1 + 1:
        return False

    col_overlap = interval_overlap(qc0, qc1, ac0, ac1)
    if col_overlap <= 0:
        return False

    return True


def is_model_relation_spatially_valid(q, a):
    if is_same_cell_near(q, a):
        return True, "model_same_cell_near"

    if is_left_neighbor(q, a):
        return True, "model_left_neighbor"

    if is_above_neighbor(q, a):
        return True, "model_above_neighbor"

    if is_near_section_relation(q, a):
        return True, "model_near_section"

    return False, "model_relation_too_far"

"""
判断模型预测的 q/a 是否真的在附近。
只有同 cell、左邻、上邻、紧邻 section 才允许模型补入。
远距离模型关系会被拒绝。
"""


def relation_key(qid, aid):
    return f"{qid}->{aid}"


def answer_key(rel):
    return rel.get("answer_id")


def convert_model_relations(model_rows, merged_items):
    by_key, by_text = build_merged_lookup(merged_items)

    model_relations = []
    unresolved_model_pairs = []

    seen = set()

    for row in model_rows:
        for head, tail in iter_model_pairs(row):
            q = resolve_model_item(head, by_key, by_text)
            a = resolve_model_item(tail, by_key, by_text)

            if not q or not a:
                unresolved_model_pairs.append({
                    "image_path": row.get("image_path"),
                    "head_text": text_of(head),
                    "tail_text": text_of(tail),
                    "head_bbox": bbox_of(head),
                    "tail_bbox": bbox_of(tail),
                    "resolved_head": bool(q),
                    "resolved_tail": bool(a),
                })
                continue

            if q.get("final_pred") not in ("QUESTION", "HEADER"):
                continue
            if a.get("final_pred") != "ANSWER":
                continue
            if q.get("in_subtable") or a.get("in_subtable"):
                continue

            key = relation_key(q["id"], a["id"])
            if key in seen:
                continue
            seen.add(key)

            model_relations.append({
                "question_id": q["id"],
                "question_text": text_of(q),
                "answer_id": a["id"],
                "answer_text": text_of(a),
                "source": "model_re",
                "image_path": row.get("image_path"),
                "question_logic_box": q.get("logic_box"),
                "answer_logic_box": a.get("logic_box"),
            })

    return model_relations, unresolved_model_pairs


# def build_final_re(data, model_relations):
#     rule_relations = data.get("rule_re", {}).get("relations") or []
#     rule_by_pair = {
#         relation_key(r["question_id"], r["answer_id"]): r
#         for r in rule_relations
#     }
#     rule_by_answer = {
#         r["answer_id"]: r
#         for r in rule_relations
#     }

#     model_only = []
#     conflicts = []
#     accepted_model = []

#     for mr in model_relations:
#         pair_key = relation_key(mr["question_id"], mr["answer_id"])
#         if pair_key in rule_by_pair:
#             continue

#         rr = rule_by_answer.get(mr["answer_id"])
#         if rr:
#             conflicts.append({
#                 "answer_id": mr["answer_id"],
#                 "answer_text": mr["answer_text"],
#                 "rule_relation": rr,
#                 "model_relation": mr,
#             })
#             continue

#         model_only.append(mr)
#         accepted_model.append(mr)

#     final_relations = []
#     for r in rule_relations:
#         x = dict(r)
#         x["source"] = "rule_re"
#         final_relations.append(x)
#     final_relations.extend(accepted_model)

#     linked_answer_ids = {r["answer_id"] for r in final_relations}

#     item_by_id = {x.get("id"): x for x in data.get("items", []) if x.get("id")}
#     if data.get("main_item_ids"):
#         main_ids = set(data.get("main_item_ids") or [])
#         main_items = [item_by_id[x] for x in main_ids if x in item_by_id]
#     else:
#         main_items = [x for x in data.get("items", []) if not x.get("in_subtable")]

#     still_unlinked = []
#     for item in main_items:
#         if item.get("final_pred") != "ANSWER":
#             continue
#         if item.get("id") in linked_answer_ids:
#             continue
#         still_unlinked.append({
#             "answer_id": item["id"],
#             "answer_text": text_of(item),
#             "answer_logic_box": item.get("logic_box"),
#             "answer_bbox": bbox_of(item),
#         })

#     return {
#         "name": "final_re_rule_plus_model_v1",
#         "policy": "Keep rule relations first. Add model relations only when the answer has no rule relation. Put rule/model disagreement into conflict_relations.",
#         "relation_count": len(final_relations),
#         "rule_relation_count": len(rule_relations),
#         "model_only_relation_count": len(model_only),
#         "conflict_relation_count": len(conflicts),
#         "still_unlinked_answer_count": len(still_unlinked),
#         "final_relations": final_relations,
#         "accepted_rule_relations": rule_relations,
#         "model_only_relations": model_only,
#         "conflict_relations": conflicts,
#         "still_unlinked_answers": still_unlinked,
#     }

"""
作用：

模型关系先过空间检查。
太远的模型关系进入：
rejected_model_relations
通过空间检查但和规则冲突的进入：
conflict_relations
只有通过空间检查且规则没连的，才进入：
model_only_relations
"""


def build_final_re(data, model_relations):
    item_by_id = {x.get("id"): x for x in data.get("items", []) if x.get("id")}

    rule_relations = data.get("rule_re", {}).get("relations") or []
    rule_by_pair = {
        relation_key(r["question_id"], r["answer_id"]): r
        for r in rule_relations
    }
    rule_by_answer = {
        r["answer_id"]: r
        for r in rule_relations
    }

    model_only = []
    rejected_model = []
    conflicts = []
    accepted_model = []

    for mr in model_relations:
        pair_key = relation_key(mr["question_id"], mr["answer_id"])
        if pair_key in rule_by_pair:
            continue

        q_item = item_by_id.get(mr["question_id"])
        a_item = item_by_id.get(mr["answer_id"])
        is_valid, spatial_reason = is_model_relation_spatially_valid(q_item, a_item)

        mr = dict(mr)
        mr["spatial_reason"] = spatial_reason

        if not is_valid:
            rejected_model.append(mr)
            continue

        rr = rule_by_answer.get(mr["answer_id"])
        if rr:
            conflicts.append({
                "answer_id": mr["answer_id"],
                "answer_text": mr["answer_text"],
                "rule_relation": rr,
                "model_relation": mr,
            })
            continue

        model_only.append(mr)
        accepted_model.append(mr)

    final_relations = []
    for r in rule_relations:
        x = dict(r)
        x["source"] = "rule_re"
        final_relations.append(x)
    final_relations.extend(accepted_model)

    linked_answer_ids = {r["answer_id"] for r in final_relations}

    if data.get("main_item_ids"):
        main_ids = set(data.get("main_item_ids") or [])
        main_items = [item_by_id[x] for x in main_ids if x in item_by_id]
    else:
        main_items = [x for x in data.get("items", []) if not x.get("in_subtable")]

    still_unlinked = []
    for item in main_items:
        if item.get("final_pred") != "ANSWER":
            continue
        if item.get("id") in linked_answer_ids:
            continue
        still_unlinked.append({
            "answer_id": item["id"],
            "answer_text": text_of(item),
            "answer_logic_box": item.get("logic_box"),
            "answer_bbox": bbox_of(item),
        })

    return {
        "name": "final_re_rule_plus_model_v2_spatial_guard",
        "policy": "Keep rule relations first. Add model relations only when the answer has no rule relation and the model relation is spatially valid. Put rule/model disagreement into conflict_relations. Put far model predictions into rejected_model_relations.",
        "relation_count": len(final_relations),
        "rule_relation_count": len(rule_relations),
        "model_only_relation_count": len(model_only),
        "rejected_model_relation_count": len(rejected_model),
        "conflict_relation_count": len(conflicts),
        "still_unlinked_answer_count": len(still_unlinked),
        "final_relations": final_relations,
        "accepted_rule_relations": rule_relations,
        "model_only_relations": model_only,
        "rejected_model_relations": rejected_model,
        "conflict_relations": conflicts,
        "still_unlinked_answers": still_unlinked,
    }

"""
作用：

模型关系先过空间检查。
太远的模型关系进入：
rejected_model_relations
通过空间检查但和规则冲突的进入：
conflict_relations
只有通过空间检查且规则没连的，才进入：
model_only_relations
"""



def process_file(input_file, output_file, model_rows):
    data = read_json(input_file)
    model_relations, unresolved = convert_model_relations(model_rows, data.get("items", []))
    final_re = build_final_re(data, model_relations)

    data["model_re"] = {
        "relation_count": len(model_relations),
        "relations": model_relations,
        "unresolved_pair_count": len(unresolved),
        "unresolved_pairs": unresolved[:200],
    }
    data["final_re"] = final_re

    write_json(output_file, data)

    return {
        "input": str(input_file),
        "output": str(output_file),
        "model_relation_count": len(model_relations),
        "unresolved_pair_count": len(unresolved),
        "final_relation_count": final_re["relation_count"],
        "model_only_relation_count": final_re["model_only_relation_count"],
        "conflict_relation_count": final_re["conflict_relation_count"],
        "still_unlinked_answer_count": final_re["still_unlinked_answer_count"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule_re_root", required=True)
    parser.add_argument("--model_re_infer", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    rule_re_root = Path(args.rule_re_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model_rows = load_model_re_infer(args.model_re_infer)

    results = []
    for input_file in sorted(rule_re_root.glob("logical_table_*_ser_merged.json")):
        output_file = output_root / input_file.name
        results.append(process_file(input_file, output_file, model_rows))

    write_json(output_root / "manifest.json", {
        "rule_re_root": str(rule_re_root),
        "model_re_infer": str(args.model_re_infer),
        "output_root": str(output_root),
        "script": "post_compare_rule_re_with_model_re.py",
        "results": results,
    })

    print(f"processed {len(results)} files")
    for r in results:
        print(
            f"{Path(r['output']).name}: "
            f"model={r['model_relation_count']} "
            f"model_only={r['model_only_relation_count']} "
            f"conflict={r['conflict_relation_count']} "
            f"unlinked={r['still_unlinked_answer_count']} "
            f"unresolved_pairs={r['unresolved_pair_count']}"
        )


if __name__ == "__main__":
    main()