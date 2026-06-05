"""
作用：

不改 PaddleOCR 官方 predict_kie_token_ser_re.py
复用现有 SerPredictor
支持你之前加的 --ocr_info_dir
跑 SER + RE
修复 RE 部署模型输入顺序问题：按 input name 喂入，而不是按列表顺序硬塞
"""


import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import json
import numpy as np
import time

import tools.infer.utility as utility
from tools.infer_kie_token_ser_re import make_input
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.visual import draw_re_results
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppstructure.utility import parse_args
from ppstructure.kie.predict_kie_token_ser import SerPredictor

logger = get_logger()


"""
作用：

只允许同一个 fragment 内建立 RE 候选。
question 可以连：
同 cell 内右侧/下侧 answer
右边紧邻 answer
下方紧邻 answer
不再让 question 连远处 answer。
"""
def _logic_box(item):
    lb = item.get("logic_box")
    if isinstance(lb, list) and len(lb) == 4:
        return lb
    return None


def _bbox(item):
    b = item.get("logical_bbox") or item.get("virtual_bbox") or item.get("bbox")
    if isinstance(b, list) and len(b) == 4:
        return b
    return None


def _interval_overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _same_fragment(q, a):
    return (
        q.get("group_id") == a.get("group_id")
        and q.get("fragment_index") == a.get("fragment_index")
    )


def _same_cell_or_split_pair(q, a):
    if q.get("cell_id") != a.get("cell_id"):
        return False

    qid = q.get("id") or ""
    aid = a.get("id") or ""
    if qid.endswith("_key") and aid.endswith("_value") and qid[:-4] == aid[:-6]:
        return True

    qb = _bbox(q)
    ab = _bbox(a)
    if not qb or not ab:
        return False

    q_left, q_top, q_right, q_bottom = qb
    a_left, a_top, a_right, a_bottom = ab

    y_overlap = _interval_overlap(q_top, q_bottom, a_top, a_bottom)
    x_overlap = _interval_overlap(q_left, q_right, a_left, a_right)

    q_is_left = q_right <= a_left + 3 and y_overlap > 0
    q_is_above = q_bottom <= a_top + 3 and x_overlap > 0

    return q_is_left or q_is_above


def _right_neighbor(q, a):
    qlb = _logic_box(q)
    alb = _logic_box(a)
    if not qlb or not alb:
        return False

    qr0, qr1, qc0, qc1 = qlb
    ar0, ar1, ac0, ac1 = alb

    if qc1 + 1 != ac0:
        return False

    return _interval_overlap(qr0, qr1, ar0, ar1) > 0


def _below_neighbor(q, a):
    qlb = _logic_box(q)
    alb = _logic_box(a)
    if not qlb or not alb:
        return False

    qr0, qr1, qc0, qc1 = qlb
    ar0, ar1, ac0, ac1 = alb

    if qr1 + 1 != ar0:
        return False

    return _interval_overlap(qc0, qc1, ac0, ac1) > 0


def _nearby_question_answer(q, a):
    if not _same_fragment(q, a):
        return False

    if _same_cell_or_split_pair(q, a):
        return True

    if _right_neighbor(q, a):
        return True

    if _below_neighbor(q, a):
        return True

    return False

"""
作用：

只允许同一个 fragment 内建立 RE 候选。
question 可以连：
同 cell 内右侧/下侧 answer
右边紧邻 answer
下方紧邻 answer
不再让 question 连远处 answer。
"""


"""
作用：

保留官方 make_input() 的实体构造方式。
只改 RE candidate 构造。
原来：
所有 QUESTION x 所有 ANSWER
现在：
QUESTION x 附近 ANSWER
"""

def make_input_nearby(ser_inputs, ser_results):
    entities_labels = {"HEADER": 0, "QUESTION": 1, "ANSWER": 2}
    batch_size, max_seq_len = ser_inputs[0].shape[:2]
    raw_entities = ser_inputs[8][0]
    ser_results_one = ser_results[0]
    assert len(raw_entities) == len(ser_results_one)

    start = []
    end = []
    label = []
    entity_idx_dict = {}

    for i, (res, entity) in enumerate(zip(ser_results_one, raw_entities)):
        if res["pred"] == "O":
            continue

        if res["pred"] not in entities_labels:
            continue

        entity_idx_dict[len(start)] = i
        start.append(entity["start"])
        end.append(entity["end"])
        label.append(entities_labels[res["pred"]])

    entities = np.full([max_seq_len + 1, 3], fill_value=-1, dtype=np.int64)
    entities[0, 0] = len(start)
    entities[1:len(start) + 1, 0] = start
    entities[0, 1] = len(end)
    entities[1:len(end) + 1, 1] = end
    entities[0, 2] = len(label)
    entities[1:len(label) + 1, 2] = label

    head = []
    tail = []

    for head_entity_idx, head_original_idx in entity_idx_dict.items():
        q = ser_results_one[head_original_idx]
        if q.get("pred") != "QUESTION":
            continue

        for tail_entity_idx, tail_original_idx in entity_idx_dict.items():
            a = ser_results_one[tail_original_idx]
            if a.get("pred") != "ANSWER":
                continue

            if _nearby_question_answer(q, a):
                head.append(head_entity_idx)
                tail.append(tail_entity_idx)

    relations = np.full([len(head) + 1, 2], fill_value=-1, dtype=np.int64)
    relations[0, 0] = len(head)
    relations[1:len(head) + 1, 0] = head
    relations[0, 1] = len(tail)
    relations[1:len(tail) + 1, 1] = tail

    entities = np.expand_dims(entities, axis=0)
    entities = np.repeat(entities, batch_size, axis=0)

    relations = np.expand_dims(relations, axis=0)
    relations = np.repeat(relations, batch_size, axis=0)

    ser_inputs = ser_inputs[:5] + [entities, relations]

    entity_idx_dict_batch = []
    for _ in range(batch_size):
        entity_idx_dict_batch.append(entity_idx_dict)

    return ser_inputs, entity_idx_dict_batch
"""
作用：

保留官方 make_input() 的实体构造方式。
只改 RE candidate 构造。
原来：
所有 QUESTION x 所有 ANSWER
现在：
QUESTION x 附近 ANSWER
"""




class MineruSerRePredictor(object):
    def __init__(self, args):
        self.use_visual_backbone = args.use_visual_backbone
        self.ser_engine = SerPredictor(args)

        if args.re_model_dir is None:
            raise ValueError("--re_model_dir is required for SER+RE inference.")

        postprocess_params = {"name": "VQAReTokenLayoutLMPostProcess"}
        self.postprocess_op = build_post_process(postprocess_params)

        self.predictor, self.input_tensor, self.output_tensors, self.config = (
            utility.create_predictor(args, "re", logger)
        )
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()

        logger.info("RE predictor input names: {}".format(self.input_names))
        logger.info("RE predictor output names: {}".format(self.output_names))

    def build_re_input_values(self, re_input):
        if self.use_visual_backbone:
            return {
                "input_ids": re_input[0].astype("int64"),
                "bbox": re_input[1].astype("int64"),
                "attention_mask": re_input[2].astype("int64"),
                "token_type_ids": re_input[3].astype("int64"),
                "image": re_input[4].astype("float32"),
                "entities": re_input[5].astype("int64"),
                "relations": re_input[6].astype("int64"),
            }

        return {
            "input_ids": re_input[0].astype("int64"),
            "bbox": re_input[1].astype("int64"),
            "attention_mask": re_input[2].astype("int64"),
            "token_type_ids": re_input[3].astype("int64"),
            "entities": re_input[4].astype("int64"),
            "relations": re_input[5].astype("int64"),
        }

    def match_input_key(self, input_name, input_values):
        name = input_name.lower()

        # Put longer/specific names first so token_type_ids is not confused with input_ids.
        ordered_keys = [
            "attention_mask",
            "token_type_ids",
            "input_ids",
            "relations",
            "entities",
            "image",
            "bbox",
        ]

        for key in ordered_keys:
            if key in input_values and key in name:
                return key

        return None

    # def copy_re_inputs_by_name(self, re_input):
    #     input_values = self.build_re_input_values(re_input)

    #     for name, handle in zip(self.input_names, self.input_tensor):
    #         key = self.match_input_key(name, input_values)
    #         if key is None:
    #             raise ValueError(
    #                 "Cannot map RE input name '{}'. Available keys: {}".format(
    #                     name, list(input_values.keys())
    #                 )
    #             )

    #         value = input_values[key]
    #         logger.info(
    #             "feed RE input: {} <= {}, shape={}, dtype={}".format(
    #                 name, key, value.shape, value.dtype
    #             )
    #         )
    #         handle.copy_from_cpu(value)


    """
    作用：

    不再依赖 x_0/x_1 这种无意义 input name。
    如果 RE 模型只有 6 个输入，就自动按这个顺序喂：
    input_ids, bbox, attention_mask, token_type_ids, entities, relations
    自动跳过 image，避免把 float32 image 喂给 entities/relations。
    仍然保留命令里的：
    --use_visual_backbone=True
    因为 SER 阶段之前就是这么跑通的；这里只是在 RE 阶段根据模型输入数量自动丢掉 image。
    """
    
    def cast_re_input_value(self, value, slot_name):
        if slot_name == "image":
            return value.astype("float32")
        return value.astype("int64")

    def copy_re_inputs_by_name(self, re_input):
        # Full RE input from make_input:
        # 0 input_ids, 1 bbox, 2 attention_mask, 3 token_type_ids,
        # 4 image, 5 entities, 6 relations
        full_slots = [
            "input_ids",
            "bbox",
            "attention_mask",
            "token_type_ids",
            "image",
            "entities",
            "relations",
        ]

        # Some exported RE models do not use visual backbone, so their input
        # count is 6 and the image input must be dropped only for RE.
        if len(self.input_tensor) == 6 and len(re_input) == 7:
            feed_indices = [0, 1, 2, 3, 5, 6]
        elif len(self.input_tensor) == len(re_input):
            feed_indices = list(range(len(re_input)))
        else:
            raise ValueError(
                "Cannot align RE inputs: model_input_count={}, re_input_count={}, "
                "input_names={}".format(
                    len(self.input_tensor), len(re_input), self.input_names
                )
            )

        for handle, input_name, src_idx in zip(
            self.input_tensor,
            self.input_names,
            feed_indices,
        ):
            slot_name = full_slots[src_idx]
            value = self.cast_re_input_value(re_input[src_idx], slot_name)

            logger.info(
                "feed RE input: {} <= {}[{}], shape={}, dtype={}".format(
                    input_name,
                    slot_name,
                    src_idx,
                    value.shape,
                    value.dtype,
                )
            )

            handle.copy_from_cpu(value)

    def get_relation_candidate_count(self, re_input):
        relations = re_input[-1]
        try:
            return int(relations[0, 0, 0])
        except Exception:
            return 0

    def __call__(self, img, image_file=None, ocr_info_path=None):
        starttime = time.time()

        ser_results, ser_inputs, ser_elapse = self.ser_engine(
            img,
            image_file=image_file,
            ocr_info_path=ocr_info_path,
        )

        # re_input, entity_idx_dict_batch = make_input(ser_inputs, ser_results)

        # if self.use_visual_backbone is False:
        #     re_input.pop(4)

        # self.copy_re_inputs_by_name(re_input)

        # self.predictor.run()
        
        # re_input, entity_idx_dict_batch = make_input(ser_inputs, ser_results)
        # 调用我们自己做的 make_input_nearby 函数，question只找附近的answer进行匹配
        re_input, entity_idx_dict_batch = make_input_nearby(ser_inputs, ser_results)

        if self.use_visual_backbone is False:
            re_input.pop(4)

        relation_candidate_count = self.get_relation_candidate_count(re_input)
        if relation_candidate_count <= 0:
            logger.info(
                "skip RE because no QUESTION-ANSWER candidates: image_file={}".format(
                    image_file
                )
            )
            elapse = time.time() - starttime
            return [[]], elapse

        self.copy_re_inputs_by_name(re_input)

        self.predictor.run()

        outputs = []
        for output_tensor in self.output_tensors:
            outputs.append(output_tensor.copy_to_cpu())

        preds = {
            "hidden_states": outputs[0],
            "loss": outputs[1],
            "pred_relations": outputs[2],
        }

        post_result = self.postprocess_op(
            preds,
            ser_results=ser_results,
            entity_idx_dict_batch=entity_idx_dict_batch,
        )

        elapse = time.time() - starttime
        return post_result, elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    predictor = MineruSerRePredictor(args)

    os.makedirs(args.output, exist_ok=True)
    infer_path = os.path.join(args.output, "infer.txt")

    count = 0
    total_time = 0

    with open(infer_path, mode="w", encoding="utf-8") as f_w:
        for image_file in image_file_list:
            img, flag, _ = check_and_read(image_file)
            if not flag:
                img = cv2.imread(image_file)

            if img is None:
                logger.info("error in loading image: {}".format(image_file))
                continue

            re_res, elapse = predictor(img, image_file=image_file)
            re_res = re_res[0]

            res_str = "{}\t{}\n".format(
                image_file,
                json.dumps({"ocr_info": re_res}, ensure_ascii=False),
            )
            f_w.write(res_str)

            img_res = draw_re_results(
                image_file,
                re_res,
                font_path=args.vis_font_path,
            )
            img_save_path = os.path.join(
                args.output,
                os.path.splitext(os.path.basename(image_file))[0] + "_ser_re.jpg",
            )
            cv2.imwrite(img_save_path, img_res)

            logger.info("save vis result to {}".format(img_save_path))
            logger.info("Predict time of {}: {}".format(image_file, elapse))

            if count > 0:
                total_time += elapse
            count += 1

    if count > 1:
        logger.info("Avg predict time: {}".format(total_time / (count - 1)))

    logger.info("save infer result to {}".format(infer_path))


if __name__ == "__main__":
    main(parse_args())