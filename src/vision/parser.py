# TODO for actions <----> cords must be multiplied by 10 because we divided it by 10

import os
os.environ["OMP_NUM_THREADS"] = "6"

from paddleocr import PaddleOCR
import numpy as np
import json
from src import utils

MIN_CONF: float = 0.75
MAX_GARBAGE_RATIO: float = 0.4
MAX_GARBLED_LENGTH: int = 35

class parser_engine:
    def __init__(self, debug: bool = False):
        self.ocr_engine = PaddleOCR(
            lang='en',
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_name='en_PP-OCRv5_mobile_rec',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=True,
            cpu_threads=6,
        )
        self.ocr_engine.predict(np.zeros((64, 64, 3), dtype=np.uint8))
        self.PATH = utils.path()
        self.DUMP_PATH = self.PATH.find_and_create_temp()
        self.DEBUG = debug

    def get_raw_screen_data(self, image) -> list[dict]:
        """
        Map out detected strings from screen and return raw data

        - params: screenshot as a PIL.Image or numpy array
        - return: list of {"text", "pos": [x, y], "conf"} dicts
        - example: {'text': '2026-04-22', 'pos': [1870, 1062], 'conf': 1.0}
        """
        img_array = np.array(image)
        results = self.ocr_engine.predict(img_array)

        if self.DEBUG:
            with open(os.path.join(self.DUMP_PATH, "recent_ocr_dump.txt"), "w", encoding="utf-8") as f:
                f.write(str(results))

        ui_elements = []
        if not results:
            return ui_elements

        result = results[0]
        for text, confidence, corners in zip(
            result.get("rec_texts", []),
            result.get("rec_scores", []),
            result.get("rec_polys", [])
        ):
            center_x = int(sum(p[0] for p in corners) / 4)
            center_y = int(sum(p[1] for p in corners) / 4)
            ui_elements.append({
                "text": text,
                "pos": [center_x, center_y],
                "conf": round(float(confidence), 2),
            })

        return ui_elements

    def is_garbage(self, text: str) -> bool:
        """
        Detect garbled OCR output.
        Checks symbol ratio AND catches long garbled status bar strings.
        """
        if not text:
            return True

        # too long with too many spaces = merged status bar elements
        if len(text) > MAX_GARBLED_LENGTH and text.count(' ') > 4:
            return True

        symbols = [c for c in text if not c.isalnum() and not c.isspace()]
        ratio = len(symbols) / len(text)
        return ratio > MAX_GARBAGE_RATIO

    def filter_elements(self, elements: list[dict]) -> list[dict]:
        """
        Pass the raw data through the filter
        Filter out the low confidence nodes and empty spots, only need strings

        - params: raw data
        - return: filtered data
        """
        filtered = []

        for item in elements:
            if item["conf"] < MIN_CONF:
                continue

            text = item["text"].strip()

            if len(text) < 2:
                continue

            if not any(c.isalnum() for c in text):
                continue

            if self.is_garbage(text):  # check whole string, not chars
                continue

            filtered.append({**item, "text": text})  # store stripped text

        return filtered

    def compress_with_regions(self, elements: list[dict], screen_w=1920, screen_h=1080) -> str:
        """
        Final layer to prepare the data for the AI
        Compress all the data and sort into regions to help the AI understand screen layout better
        This will further maximise efficiency for less token usage

        - params: none
        - return: pure compresed list
        - example: Save Button@12,5 <--> means Save Button element in vector (120, 50)
        """

        def get_zone(x, y) -> str:
            if y < screen_h * 0.08:
                return "topbar"
            elif y > screen_h * 0.92:
                return "taskbar"
            elif x < screen_w * 0.15:
                return "sidebar"
            else:
                return "main"

        zones: dict[str, list[str]] = {}
        for item in elements:
            x, y = item["pos"]
            zone = get_zone(x, y)
            if zone not in zones:
                zones[zone] = []
            # clean simple format, no wrapper noise
            zones[zone].append(f"{item['text']}@{x//10},{y//10}")

        lines = []
        for zone, items in zones.items():
            lines.append(f"-- [{zone}] --")
            lines.extend(items)

        return "\n".join(lines)

    def get_efficient_screen_handle(self, image) -> str:
        """
        This will return filtered, compressed and organised data
        This will help:
            - delete bloat
            - reduce token usage
            - let ai get a grasp of the layout
        
        - params: image of the screen
        - return: highly effecient compressed data set ready to send to the AI
        """
        
        raw = self.get_raw_screen_data(image)
        filtered = self.filter_elements(raw)
        result = self.compress_with_regions(filtered)

        if self.DEBUG:
            with open(os.path.join(self.DUMP_PATH, "recent_final_result.txt"), "w", encoding="utf-8") as f:
                f.write(result)

        return result