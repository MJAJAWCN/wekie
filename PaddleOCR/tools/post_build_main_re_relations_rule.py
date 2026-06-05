"""
作用：只做规则法第一轮 RE。它会：

读取 subtable_marked 后的结果
剔除 in_subtable=True 的子表 item
只在主表里找 QUESTION -> ANSWER
同 cell 最高优先级
左边 question 和上边 question 同一优先级竞争
找不到的 answer 放进 unlinked_answers
不改原文件，输出到新目录
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


def logic_box(item):
    lb = item.get("logic_box")
    if isinstance(lb, list) and len(lb) == 4:
        return lb
    return None


def bbox(item):
    b = item.get("logical_bbox") or item.get("virtual_bbox") or item.get("bbox")
    if isinstance(b, list) and len(b) == 4:
        return b
    return None


def center_x(item):
    b = bbox(item)
    if not b:
        return 0
    return (b[0] + b[2]) / 2.0


def center_y(item):
    b = bbox(item)
    if not b:
        return 0
    return (b[1] + b[3]) / 2.0


"""
作用：

is_split_key_value_pair()：保护已经拆好的 _key -> _value，这类仍然可以直接连。
is_before_in_same_cell()：同 cell 里也要判断 question 是否真的在 answer 左侧或上侧。
"""

def is_split_key_value_pair(q, a):
    qid = q.get("id") or ""
    aid = a.get("id") or ""
    if not qid.endswith("_key") or not aid.endswith("_value"):
        return False

    return qid[:-4] == aid[:-6]


def is_before_in_same_cell(q, a, tolerance=3):
    qb = bbox(q)
    ab = bbox(a)
    if not qb or not ab:
        return False

    q_left, q_top, q_right, q_bottom = qb
    a_left, a_top, a_right, a_bottom = ab

    q_center_x = (q_left + q_right) / 2.0
    q_center_y = (q_top + q_bottom) / 2.0
    a_center_x = (a_left + a_right) / 2.0
    a_center_y = (a_top + a_bottom) / 2.0

    y_overlap = interval_overlap(q_top, q_bottom, a_top, a_bottom)
    x_overlap = interval_overlap(q_left, q_right, a_left, a_right)

    q_is_left = q_right <= a_left + tolerance and y_overlap > 0
    q_is_above = q_bottom <= a_top + tolerance and x_overlap > 0

    if q_is_left or q_is_above:
        return True

    # 有些 OCR 框会轻微重叠，用中心点再兜底一次，但仍要求方向明确。
    mostly_left = q_center_x < a_center_x and y_overlap > 0 and q_right <= a_right
    mostly_above = q_center_y < a_center_y and x_overlap > 0 and q_bottom <= a_bottom

    return mostly_left or mostly_above

def interval_overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def covers(a0, a1, b0, b1):
    return a0 <= b0 and a1 >= b1

# 同 fragment 里才能连，判断 question 和 answer 是否属于同一个逻辑表 group、同一个 fragment。
def same_fragment(q, a):
    return (
        q.get("group_id") == a.get("group_id")
        and q.get("fragment_index") == a.get("fragment_index")
    )

# def same_cell_candidate(q, a):
#     if q.get("cell_id") != a.get("cell_id"):
#         return None

#     qlb = logic_box(q)
#     alb = logic_box(a)
#     if qlb and alb and qlb != alb:
#         return None

#     dist = abs(center_x(a) - center_x(q)) + abs(center_y(a) - center_y(q)) * 0.2

#     return {
#         "question_id": q["id"],
#         "answer_id": a["id"],
#         "reason": "same_cell_question_answer",
#         "direction": "same_cell",
#         "confidence": 1000 - dist * 0.01,
#         "distance": dist,
#         "question_text": text_of(q),
#         "answer_text": text_of(a),
#         "question_logic_box": qlb,
#         "answer_logic_box": alb,
#     }

"""
作用：

以前：同 cell 就能连。
现在：同 cell 也必须满足：
是明确 _key -> _value，或者
question 的 bbox 在 answer 左侧/上侧。
这样可以防止 其他： -> GJB150.3A-2009 这种右侧/后出现的字段误连。
进箱操作人员： -> 赵操作 仍然会保留，因为 进箱操作人员： 在 answer 左侧
"""

def same_cell_candidate(q, a):
    # 添加一处代码
    # 同 cell 关系也必须在同一个 fragment 内。
    if not same_fragment(q, a):
        return None
    
    if q.get("cell_id") != a.get("cell_id"):
        return None

    qlb = logic_box(q)
    alb = logic_box(a)
    if qlb and alb and qlb != alb:
        return None

    split_pair = is_split_key_value_pair(q, a)
    before_answer = is_before_in_same_cell(q, a)

    if not split_pair and not before_answer:
        return None

    dist = abs(center_x(a) - center_x(q)) + abs(center_y(a) - center_y(q)) * 0.2

    reason = "same_cell_split_key_value" if split_pair else "same_cell_before_question"

    return {
        "question_id": q["id"],
        "answer_id": a["id"],
        "reason": reason,
        "direction": "same_cell",
        "confidence": 1000 - dist * 0.01,
        "distance": dist,
        "question_text": text_of(q),
        "answer_text": text_of(a),
        "question_logic_box": qlb,
        "answer_logic_box": alb,
    }

def left_candidate(q, a):
    # 添加一处代码
    # 左邻关系必须同 fragment，防止 CAS号 -> 操作控制正文 这种跨 fragment 假相邻。
    if not same_fragment(q, a):
        return None
    
    qlb = logic_box(q)
    alb = logic_box(a)
    if not qlb or not alb:
        return None

    qr0, qr1, qc0, qc1 = qlb
    ar0, ar1, ac0, ac1 = alb

    if qc1 >= ac0:
        return None

    row_overlap = interval_overlap(qr0, qr1, ar0, ar1)
    if row_overlap <= 0:
        return None

    direct_col = qc1 + 1 == ac0
    row_cover = covers(qr0, qr1, ar0, ar1)
    row_exact = qr0 == ar0 and qr1 == ar1

    if not direct_col:
        return None

    dist = ac0 - qc1
    quality = 0
    if row_exact:
        quality += 30
    elif row_cover:
        quality += 25
    else:
        quality += 10

    return {
        "question_id": q["id"],
        "answer_id": a["id"],
        "reason": "left_adjacent_question",
        "direction": "left",
        "confidence": 800 + quality - dist,
        "distance": dist,
        "question_text": text_of(q),
        "answer_text": text_of(a),
        "question_logic_box": qlb,
        "answer_logic_box": alb,
    }


def above_candidate(q, a):
    # 添加一处代码
    # 上邻关系也先限制在同 fragment 内。
    # 跨页长正文这种不要靠普通 above 规则猜，后面应该由 long_text_fix 里已经保护好的 key 或专门跨 fragment 规则处理。
    if not same_fragment(q, a):
        return None
    
    qlb = logic_box(q)
    alb = logic_box(a)
    if not qlb or not alb:
        return None

    qr0, qr1, qc0, qc1 = qlb
    ar0, ar1, ac0, ac1 = alb

    if qr1 >= ar0:
        return None

    col_overlap = interval_overlap(qc0, qc1, ac0, ac1)
    if col_overlap <= 0:
        return None

    direct_row = qr1 + 1 == ar0
    col_cover = covers(qc0, qc1, ac0, ac1)
    col_exact = qc0 == ac0 and qc1 == ac1

    if not direct_row:
        return None

    dist = ar0 - qr1
    quality = 0
    if col_exact:
        quality += 30
    elif col_cover:
        quality += 25
    else:
        quality += 10

    return {
        "question_id": q["id"],
        "answer_id": a["id"],
        "reason": "above_adjacent_question",
        "direction": "above",
        "confidence": 800 + quality - dist,
        "distance": dist,
        "question_text": text_of(q),
        "answer_text": text_of(a),
        "question_logic_box": qlb,
        "answer_logic_box": alb,
    }


def build_rule_relations(data):
    item_by_id = {x.get("id"): x for x in data.get("items", []) if x.get("id")}

    if data.get("main_item_ids"):
        main_ids = set(data.get("main_item_ids") or [])
        items = [item_by_id[x] for x in main_ids if x in item_by_id]
    else:
        items = [x for x in data.get("items", []) if not x.get("in_subtable")]

    # questions = [x for x in items if x.get("final_pred") == "QUESTION"]
    # answers = [x for x in items if x.get("final_pred") == "ANSWER"]
    
    """
    原来规则 RE 只认 final_pred == QUESTION 的字段名。
    现在额外把 O 里“看起来像字段名”的 item 也临时当作 question 候选。
    当前只用了最保守条件：文本以 : 或 ： 结尾。
    这样 进箱操作人员： 虽然 SER 标成了 O，RE 规则仍然能把它作为 question，连出：
    进箱操作人员： -> 赵操作
    注意：这段只是 RE 阶段临时候选，不会修改原 item 的 final_pred，也不会污染前面的 SER 结果。子表 item 已经被剔除，所以不会影响子表导出。
    """
    
    def is_question_like_o(item):
        if item.get("final_pred") != "O":
            return False

        text = text_of(item).strip()
        if not text:
            return False

        # if text.endswith((":", "：")):
        #     return True
        if text.endswith((":", "\uff1a")):
            return True

        return False


    questions = [
        x for x in items
        if x.get("final_pred") == "QUESTION" or is_question_like_o(x)
    ]
    answers = [x for x in items if x.get("final_pred") == "ANSWER"]

    relations = []
    unlinked_answers = []
    ambiguous_answers = []

    for answer in sorted(answers, key=lambda x: (logic_box(x) or [999999] * 4, center_y(x), center_x(x), x.get("id"))):
        candidates = []

        for question in questions:
            if question.get("id") == answer.get("id"):
                continue

            c = same_cell_candidate(question, answer)
            if c:
                candidates.append(c)
                continue

            c = left_candidate(question, answer)
            if c:
                candidates.append(c)

            c = above_candidate(question, answer)
            if c:
                candidates.append(c)

        candidates.sort(key=lambda x: (-x["confidence"], x["distance"], x["question_id"]))

        if not candidates:
            unlinked_answers.append({
                "answer_id": answer["id"],
                "answer_text": text_of(answer),
                "answer_logic_box": logic_box(answer),
                "answer_bbox": bbox(answer),
            })
            continue

        best = candidates[0]
        alternatives = candidates[1:5]

        relation = {
            "question_id": best["question_id"],
            "question_text": best["question_text"],
            "answer_id": best["answer_id"],
            "answer_text": best["answer_text"],
            "reason": best["reason"],
            "direction": best["direction"],
            "confidence": round(best["confidence"], 4),
            "question_logic_box": best["question_logic_box"],
            "answer_logic_box": best["answer_logic_box"],
            "alternative_candidates": alternatives,
        }

        if alternatives and abs(best["confidence"] - alternatives[0]["confidence"]) < 1e-6:
            relation["ambiguous"] = True
            ambiguous_answers.append(relation)
        else:
            relation["ambiguous"] = False

        relations.append(relation)

    return {
        "name": "main_table_rule_re_v1",
        "description": "Rule-based QUESTION->ANSWER relations for main table items only. Subtable items are excluded.",
        "relation_count": len(relations),
        "unlinked_answer_count": len(unlinked_answers),
        "ambiguous_answer_count": len(ambiguous_answers),
        "relations": relations,
        "unlinked_answers": unlinked_answers,
        "ambiguous_answers": ambiguous_answers,
    }


def process_file(input_file, output_file):
    data = read_json(input_file)
    rule_re = build_rule_relations(data)

    data["rule_re"] = rule_re
    write_json(output_file, data)

    return {
        "input": str(input_file),
        "output": str(output_file),
        "relation_count": rule_re["relation_count"],
        "unlinked_answer_count": rule_re["unlinked_answer_count"],
        "ambiguous_answer_count": rule_re["ambiguous_answer_count"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for input_file in sorted(input_root.glob("logical_table_*_ser_merged.json")):
        output_file = output_root / input_file.name
        results.append(process_file(input_file, output_file))

    write_json(output_root / "manifest.json", {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "script": "post_build_main_re_relations_rule.py",
        "results": results,
    })

    print(f"processed {len(results)} files")
    for r in results:
        print(
            f"{Path(r['output']).name}: "
            f"relations={r['relation_count']} "
            f"unlinked={r['unlinked_answer_count']} "
            f"ambiguous={r['ambiguous_answer_count']}"
        )


if __name__ == "__main__":
    main()