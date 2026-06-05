import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import cv2
import paddle

from ppocr.utils.visual import draw_ser_results
from tools.infer_kie_token_ser import SerPredictor
from tools.program import load_config, merge_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--opt", nargs="+", default=[])
    parser.add_argument("--image", required=True)
    parser.add_argument("--ocr_info", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_items", type=int, default=120)
    parser.add_argument("--overlap_items", type=int, default=40)
    return parser.parse_args()


def parse_opt_list(opt_list):
    out = {}
    for item in opt_list or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if value == "True":
            value = True
        elif value == "False":
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        out[key] = value
    return out


def make_windows(items, max_items, overlap_items):
    if max_items <= 0:
        raise ValueError("--max_items must be > 0")
    if overlap_items >= max_items:
        raise ValueError("--overlap_items must be smaller than --max_items")

    step = max_items - overlap_items
    windows = []
    start = 0

    while start < len(items):
        end = min(start + max_items, len(items))
        windows.append((start, end, items[start:end]))
        if end >= len(items):
            break
        start += step

    return windows


def item_key(item):
    return str(item.get("id") or item.get("ocr_id") or item.get("transcription") + str(item.get("bbox")))


def pred_rank(pred):
    return {
        "HEADER": 4,
        "QUESTION": 3,
        "ANSWER": 2,
        "O": 1,
    }.get(pred, 0)


def merge_ser_results(all_window_results):
    merged = {}
    order = []

    for result in all_window_results:
        for item in result:
            key = item_key(item)
            if key not in merged:
                merged[key] = item
                order.append(key)
                continue

            old = merged[key]
            if pred_rank(item.get("pred")) > pred_rank(old.get("pred")):
                merged[key] = item

    return [merged[k] for k in order]


def main():
    args = parse_args()

    config = load_config(args.config)
    config = merge_config(config, parse_opt_list(args.opt))
    config["Global"]["infer_mode"] = True

    use_gpu = bool(config["Global"].get("use_gpu", False))
    paddle.set_device("gpu" if use_gpu else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = json.loads(Path(args.ocr_info).read_text(encoding="utf-8"))
    windows = make_windows(items, args.max_items, args.overlap_items)

    ser_engine = SerPredictor(config)
    all_results = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="kie_ser_sliding_"))

    for win_idx, (start, end, win_items) in enumerate(windows):
        tmp_ocr = tmp_dir / f"window_{win_idx}.json"
        tmp_ocr.write_text(json.dumps(win_items, ensure_ascii=False), encoding="utf-8")

        data = {
            "img_path": args.image,
            "ocr_info_path": str(tmp_ocr),
        }

        result, _ = ser_engine(data)
        result = result[0]
        all_results.append(result)
        print(f"window {win_idx}: items {start}-{end}, result={len(result)}")

    merged = merge_ser_results(all_results)

    result_json = output_dir / "ser_results.json"
    result_json.write_text(
        json.dumps({"ocr_info": merged}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    infer_txt = output_dir / "infer_results.txt"
    infer_txt.write_text(
        args.image + "\t" + json.dumps({"ocr_info": merged}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    save_img_path = output_dir / (Path(args.image).stem + "_ser_sliding.jpg")
    img_res = draw_ser_results(args.image, merged)
    cv2.imwrite(str(save_img_path), img_res)

    print(f"saved json: {result_json}")
    print(f"saved txt: {infer_txt}")
    print(f"saved image: {save_img_path}")


if __name__ == "__main__":
    main()