import os
os.environ["OMP_NUM_THREADS"] = "6"

from paddleocr import PaddleOCR
import numpy as np
import json
import src.nanocontrol.utils as utils

class parser_engine:
    def __init__(self, debug: bool = False):
        self.ocr_engine = PaddleOCR(
            lang='en',
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_name='en_PP-OCRv5_mobile_rec',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=True, # for intel graphics
            cpu_threads=6,
        )
        self.ocr_engine.predict(np.zeros((64, 64, 3), dtype=np.uint8))

        self.PATH = utils.path()
        self.DUMP_PATH = self.PATH.find_and_create_temp()
        self.DEBUG = debug

    def get_screen_string_map(self, image) -> list[dict]:
        """
        Map text elements from a screenshot into a list of UI elements
        with their screen positions.

        - param image: screenshot as a PIL.Image or numpy array
        - return: list of {"text", "pos": [x, y], "conf"} dicts
        """

        img_array = np.array(image)
        results = self.ocr_engine.predict(img_array)

        if self.DEBUG:
            with open(
                os.path.join(self.DUMP_PATH, "recent_ocr_dump.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(str(results))

        ui_elements: list[dict] = []
        if not results:
            return ui_elements

        result = results[0]
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        polys = result.get("rec_polys", [])

        for text, confidence, corners in zip(texts, scores, polys):
            center_x = int(sum(p[0] for p in corners) / 4)
            center_y = int(sum(p[1] for p in corners) / 4)

            ui_elements.append({
                "text": text,
                "pos": [center_x, center_y],
                "conf": round(float(confidence), 2),
            })

        if self.DEBUG:
            with open(
                os.path.join(self.DUMP_PATH, "recent_ocr_dump.json"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(ui_elements, indent=2, ensure_ascii=False))

        return ui_elements