from paddleocr import PaddleOCR
import numpy as np

# Single global engine — PaddleOCR is expensive to initialise
_ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

MIN_CONFIDENCE = 0.5


def get_ui_map(image, min_conf: float = MIN_CONFIDENCE) -> list[dict]:
    """
    Extract visible text elements from a screenshot.

    Returns a list of dicts sorted in reading order (top-to-bottom,
    left-to-right), each with:
      text  — the detected string
      pos   — [x, y] pixel centre of the bounding box
      box   — [x1, y1, x2, y2] tight bounding rect (for spatial reasoning)
      conf  — OCR confidence 0–1
    """
    img_array = np.array(image)
    ocr_result = _ocr_engine.ocr(img_array)

    elements = []

    if not ocr_result or not ocr_result[0]:
        return elements

    for detection in ocr_result[0]:
        corners = detection[0]
        text, confidence = detection[1]

        if confidence < min_conf:
            continue

        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]

        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        elements.append({
            "text": text,
            "pos": [cx, cy],
            "box": [x1, y1, x2, y2],
            "conf": round(float(confidence), 2),
        })

    # Reading order: row bands of ~20px height, then left-to-right within each band
    elements.sort(key=lambda e: (e["pos"][1] // 20, e["pos"][0]))

    return elements
