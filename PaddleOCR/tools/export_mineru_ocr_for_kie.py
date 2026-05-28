"""作用：调用 MinerU 的 OCR，把结果转成 LayoutXLM 需要的 ocr_info 格式。"""



import argparse
import json
import os
import sys
from pathlib import Path

import cv2


MINERU_ROOT = r"E:\wepipelineStage2\wepipeline\MinerU"
sys.path.insert(0, MINERU_ROOT)

from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR


def box_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def points_to_int(points):
    return [[int(x), int(y)] for x, y in points]

# 下面这个函数的作用是把类似 "品质部：陈序" 这种ocr识别到的一整块短文本通过python脚本规则化地分成两行，分别是 "品质部：" 和 "陈序"，并且把它们的 bbox 分开。这样可以让 KIE 模型更好地理解它们之间的关系。
def split_key_value_item(text, points, score):
    """Split short key-value OCR text like '品质部：陈序' into two OCR items."""
    if not text:
        return []

    split_chars = ["：", ":"]
    split_pos = -1
    for ch in split_chars:
        split_pos = text.find(ch)
        if split_pos > 0:
            break

    if split_pos <= 0 or split_pos >= len(text) - 1:
        return [{
            "transcription": text,
            "bbox": box_to_bbox(points),
            "points": points_to_int(points),
            "score": float(score),
        }]

    # 品质部：陈序 会继续拆，但 14:30、16:45 这类时间不会被拆。
    left_char = text[split_pos - 1]
    right_char = text[split_pos + 1]
    if left_char.isdigit() and right_char.isdigit():
        return [{
            "transcription": text,
            "bbox": box_to_bbox(points),
            "points": points_to_int(points),
            "score": float(score),
        }]
    
    key_text = text[:split_pos + 1].strip()
    value_text = text[split_pos + 1:].strip()

    if not key_text or not value_text:
        return [{
            "transcription": text,
            "bbox": box_to_bbox(points),
            "points": points_to_int(points),
            "score": float(score),
        }]

    # Avoid splitting long sentences or dense table rows.
    if len(text) > 24:
        return [{
            "transcription": text,
            "bbox": box_to_bbox(points),
            "points": points_to_int(points),
            "score": float(score),
        }]

    bbox = box_to_bbox(points)
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)

    key_len = max(1, len(key_text))
    value_len = max(1, len(value_text))
    key_ratio = key_len / float(key_len + value_len)

    split_x = int(x1 + width * key_ratio)
    split_x = max(x1 + 1, min(split_x, x2 - 1))

    key_points = [[x1, y1], [split_x, y1], [split_x, y2], [x1, y2]]
    value_points = [[split_x, y1], [x2, y1], [x2, y2], [split_x, y2]]

    return [
        {
            "transcription": key_text,
            "bbox": box_to_bbox(key_points),
            "points": points_to_int(key_points),
            "score": float(score),
        },
        {
            "transcription": value_text,
            "bbox": box_to_bbox(value_points),
            "points": points_to_int(value_points),
            "score": float(score),
        },
    ]


def export_one(ocr_model, image_path, output_dir):
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    ocr_res = ocr_model.ocr(img, det=True, rec=True)[0]

    kie_ocr_info = []
    if ocr_res:
        for item in ocr_res:
            if not item or len(item) != 2:
                continue

            points, rec = item
            if not rec or len(rec) < 2:
                continue

            text, score = rec
            if not text:
                continue

            # kie_ocr_info.append({
            #     "transcription": text,
            #     "bbox": box_to_bbox(points),
            #     "points": points_to_int(points),
            #     "score": float(score),
            # })
            kie_ocr_info.extend(split_key_value_item(text, points, score))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_ocr_info.json"
    output_path.write_text(
        json.dumps(kie_ocr_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"saved: {output_path}, count={len(kie_ocr_info)}")


def iter_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [
        p for p in input_path.rglob("*")
        if p.suffix.lower() in suffixes
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="image file or image directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lang", default="ch")
    args = parser.parse_args()

    ocr_model = PytorchPaddleOCR(lang=args.lang)

    for image_path in iter_images(args.input):
        export_one(ocr_model, image_path, args.output_dir)


if __name__ == "__main__":
    main()