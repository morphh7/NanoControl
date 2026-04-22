# NanoControl - Parser Engine
# Coords sent to Claude are divided by 10 for token efficiency
# Action executor must multiply coords by 10 to get real pixel position
# Use item["pos"] for full precision clicking

import os
import re
import json
os.environ["OMP_NUM_THREADS"] = "6"

from paddleocr import PaddleOCR
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src import utils

MIN_CONF: float          = 0.75
MAX_GARBAGE_RATIO: float = 0.4
MAX_GARBLED_LENGTH: int  = 35
ROW_Y_TOLERANCE: int     = 12


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

    # ------------------------------------------------------------------ #
    #  STEP 1 — Raw OCR                                                   #
    # ------------------------------------------------------------------ #

    def get_raw_screen_data(self, image) -> list[dict]:
        """
        Run PaddleOCR and return ALL detections including low-confidence ones.
        poly is kept so the debug visualizer can draw exact bounding boxes.

        Returns list of:
            {
                "text": str,
                "pos":  [center_x, center_y],   # full pixel coords
                "conf": float,
                "poly": list[list[int]]          # 4 corner points from paddle
            }
        """
        img_array = np.array(image)
        results = self.ocr_engine.predict(img_array)

        if self.DEBUG:
            with open(
                os.path.join(self.DUMP_PATH, "ocr_raw.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(str(results))

        ui_elements = []
        if not results:
            return ui_elements

        result = results[0]
        for text, confidence, corners in zip(
            result.get("rec_texts", []),
            result.get("rec_scores", []),
            result.get("rec_polys", []),
        ):
            center_x = int(sum(p[0] for p in corners) / 4)
            center_y = int(sum(p[1] for p in corners) / 4)
            ui_elements.append({
                "text": text,
                "pos":  [center_x, center_y],
                "conf": round(float(confidence), 2),
                "poly": [list(map(int, p)) for p in corners],
            })

        return ui_elements

    # ------------------------------------------------------------------ #
    #  STEP 2 — Filter                                                    #
    # ------------------------------------------------------------------ #

    def is_garbage(self, text: str) -> bool:
        if not text:
            return True
        if len(text) > MAX_GARBLED_LENGTH and text.count(" ") > 4:
            return True
        symbols = [c for c in text if not c.isalnum() and not c.isspace()]
        return (len(symbols) / len(text)) > MAX_GARBAGE_RATIO

    def _fix_ocr_misreads(self, text: str) -> str:
        """Fix common Paddle misreads without altering real content."""
        # "I word" where word starts lowercase = misread capital letter
        text = re.sub(r"^I ([a-z])", r"\1", text)
        # single leading underscore on what should be a dunder name
        if re.match(r"^_[^_]", text):
            text = "_" + text
        # single trailing underscore before optional file extension
        text = re.sub(r"__(\w+)_(\.\w+)$", r"__\1__\2", text)
        text = re.sub(r"__(\w+)_$",         r"__\1__",   text)
        return text

    def filter_elements(self, elements: list[dict]) -> list[dict]:
        """
        Remove noise. Preserves full pos for action executor.
        Adds pos_compressed (÷10) for Claude output.
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
            if self.is_garbage(text):
                continue

            text = self._fix_ocr_misreads(text)
            x, y = item["pos"]

            filtered.append({
                "text":           text,
                "pos":            [x, y],              # full precision — for clicking
                "pos_compressed": [x // 10, y // 10],  # coarse — for Claude only
                "conf":           item["conf"],
                "poly":           item.get("poly", []),
            })

        return filtered

    # ------------------------------------------------------------------ #
    #  STEP 3 — Row grouping                                              #
    # ------------------------------------------------------------------ #

    def group_into_rows(self, elements: list[dict]) -> list[list[dict]]:
        """
        Group elements sharing the same y-coordinate (within ROW_Y_TOLERANCE)
        and sort each row left-to-right.

        Returns list of rows — each row is a list of elements.
        The leftmost element in each row is the primary (filename, label etc).
        Remaining elements in the row are metadata (date, size, type).
        """
        if not elements:
            return []

        sorted_els = sorted(elements, key=lambda e: e["pos"][1])
        rows: list[list[dict]] = []
        current_row = [sorted_els[0]]

        for el in sorted_els[1:]:
            if abs(el["pos"][1] - current_row[0]["pos"][1]) <= ROW_Y_TOLERANCE:
                current_row.append(el)
            else:
                rows.append(sorted(current_row, key=lambda e: e["pos"][0]))
                current_row = [el]

        rows.append(sorted(current_row, key=lambda e: e["pos"][0]))
        return rows

    # ------------------------------------------------------------------ #
    #  STEP 4 — Compress & regionalise                                    #
    # ------------------------------------------------------------------ #

    def compress_with_regions(
        self,
        rows: list[list[dict]],
        screen_w: int = 1920,
        screen_h: int = 1080,
    ) -> str:
        """
        Convert grouped rows into a token-efficient regional string for Claude.

        Single-element row:    text@cx,cy
        Multi-element row:     text@cx,cy [meta1, meta2, ...]
            primary (leftmost) gets coordinate, rest go in brackets

        Regions:  topbar | sidebar | main | taskbar
        Coordinates are pos_compressed (÷10). Multiply ×10 for real pixels.
        """

        def get_zone(x: int, y: int) -> str:
            if y < screen_h * 0.08:   return "topbar"
            if y > screen_h * 0.92:   return "taskbar"
            if x < screen_w * 0.15:   return "sidebar"
            return "main"

        def format_row(row: list[dict]) -> str:
            primary = row[0]
            cx, cy  = primary["pos_compressed"]
            base    = f"{primary['text']}@{cx},{cy}"
            if len(row) == 1:
                return base
            meta = [el["text"] for el in row[1:]]
            return f"{base} [{', '.join(meta)}]"

        zones: dict[str, list[str]] = {}
        for row in rows:
            x, y = row[0]["pos"]
            zone = get_zone(x, y)
            zones.setdefault(zone, []).append(format_row(row))

        lines: list[str] = []
        for zone, entries in zones.items():
            lines.append(f"-- [{zone}] --")
            lines.extend(entries)

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  DEBUG — OCR Visualizer                                             #
    # ------------------------------------------------------------------ #

    def debug_draw_ocr(self, image, save_path: str = None) -> Image.Image:
        """
        Draw ALL OCR detections on the image with colour-coded confidence.
        Shows what Paddle found AND what your filter is dropping.

        Colour coding:
            GREEN  >= 0.90  — high confidence, passes filter
            YELLOW 0.75-0.89 — medium confidence, passes filter
            RED    < 0.75   — filtered out (low confidence)
            BLUE            — passes confidence but flagged as garbage

        Label format:  "detected text"  conf  [STATUS]

        Usage:
            img = Image.open("screenshot.png")

            # save to file
            parser.debug_draw_ocr(img, save_path="debug.png")

            # or get PIL image back and show it
            annotated = parser.debug_draw_ocr(img)
            annotated.show()

            # or just enable debug=True and it auto-saves every parse
            parser = parser_engine(debug=True)
            parser.get_efficient_screen_handle(img)
            # → saved to DUMP_PATH/ocr_visual.png

        - param image:     PIL.Image or numpy array
        - param save_path: optional path to save the annotated image as PNG
        - return:          annotated PIL.Image
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        annotated = image.copy().convert("RGB")
        draw      = ImageDraw.Draw(annotated)

        try:
            font       = ImageFont.truetype("arial.ttf", 13)
            font_small = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            font       = ImageFont.load_default()
            font_small = font

        raw = self.get_raw_screen_data(image)

        for item in raw:
            conf   = item["conf"]
            text   = item["text"].strip()
            poly   = item.get("poly", [])
            cx, cy = item["pos"]

            # colour + status label based on confidence and garbage check
            if conf < MIN_CONF:
                color    = (220, 50,  50)    # red
                label_bg = (160, 20,  20)
                status   = "LOW"
            elif self.is_garbage(text):
                color    = (50,  120, 220)   # blue
                label_bg = (20,  70,  160)
                status   = "GARB"
            elif conf >= 0.90:
                color    = (50,  200, 80)    # green
                label_bg = (20,  140, 40)
                status   = "OK"
            else:
                color    = (220, 200, 50)    # yellow
                label_bg = (160, 140, 20)
                status   = "MED"

            # draw bounding polygon
            if len(poly) >= 4:
                pts = [(p[0], p[1]) for p in poly]
                draw.polygon(pts, outline=color)
                draw.polygon([(p[0] + 1, p[1] + 1) for p in pts], outline=color)
            else:
                # fallback cross if no polygon
                draw.line((cx - 6, cy, cx + 6, cy), fill=color, width=2)
                draw.line((cx, cy - 6, cx, cy + 6), fill=color, width=2)

            # label: truncate long text so it doesn't overflow
            label = f'"{text[:20]}" {conf:.2f} [{status}]'

            try:
                bbox = draw.textbbox((cx, cy - 18), label, font=font_small)
                # keep label inside image bounds
                if bbox[0] < 0:
                    bbox = (0, bbox[1], bbox[2] - bbox[0], bbox[3])
                draw.rectangle(bbox, fill=label_bg)
            except AttributeError:
                pass  # older Pillow — skip background rect

            draw.text((max(cx, 2), cy - 18), label, fill=(255, 255, 255), font=font_small)

        # legend — top left corner
        legend_items = [
            ("GREEN  >= 0.90  passes",          (50,  200, 80)),
            ("YELLOW 0.75-0.89 passes",          (220, 200, 50)),
            ("RED    < 0.75   filtered",         (220, 50,  50)),
            ("BLUE   garbage  filtered",         (50,  120, 220)),
        ]
        lx, ly = 10, 10
        for leg_text, leg_color in legend_items:
            draw.rectangle((lx, ly, lx + 12, ly + 12), fill=leg_color)
            draw.text((lx + 16, ly), leg_text, fill=(255, 255, 255), font=font_small)
            ly += 16

        if save_path:
            annotated.save(save_path)
            print(f"[debug] OCR visual saved → {save_path}")

        return annotated

    def debug_print_stats(self, image) -> None:
        """
        Print a breakdown of what was detected, what passed, and what was dropped.
        Run this to quickly diagnose why something is missing from the output.

        Usage:
            parser.debug_print_stats(screenshot)
        """
        raw      = self.get_raw_screen_data(image)
        filtered = self.filter_elements(raw)

        low_conf = [i for i in raw if i["conf"] < MIN_CONF]
        garbage  = [
            i for i in raw
            if i["conf"] >= MIN_CONF and self.is_garbage(i["text"].strip())
        ]

        print(f"\n{'=' * 52}")
        print(f"  OCR STATS")
        print(f"{'=' * 52}")
        print(f"  Total detected:      {len(raw)}")
        print(f"  Passed filter:       {len(filtered)}")
        print(f"  Dropped (low conf):  {len(low_conf)}")
        print(f"  Dropped (garbage):   {len(garbage)}")
        print(f"{'=' * 52}")

        if low_conf:
            print(f"\n  LOW CONFIDENCE — filtered out:")
            for i in sorted(low_conf, key=lambda x: x["conf"]):
                print(f"    [{i['conf']:.2f}]  \"{i['text']}\"  @ {i['pos']}")

        if garbage:
            print(f"\n  GARBAGE TEXT — filtered out:")
            for i in garbage:
                print(f"    [{i['conf']:.2f}]  \"{i['text']}\"  @ {i['pos']}")

        print(f"\n  PASSED:")
        for i in filtered:
            print(f"    [{i['conf']:.2f}]  \"{i['text']}\"  @ {i['pos']}")
        print()

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def get_efficient_screen_handle(self, image) -> str:
        """
        Full pipeline: raw OCR → filter → row-group → compress → regionalise.
        Returns a token-efficient string ready to send to Claude.
        ~300-500 tokens for a typical 1080p desktop.

        When debug=True also saves:
            ocr_raw.txt        — raw paddle output
            ocr_final.txt      — final string sent to Claude
            ocr_final.json     — structured data with full coords (for action executor)
            ocr_visual.png     — annotated screenshot showing all detections
        """
        raw      = self.get_raw_screen_data(image)
        filtered = self.filter_elements(raw)
        rows     = self.group_into_rows(filtered)
        result   = self.compress_with_regions(rows)

        if self.DEBUG:
            with open(
                os.path.join(self.DUMP_PATH, "ocr_final.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(result)

            with open(
                os.path.join(self.DUMP_PATH, "ocr_final.json"), "w", encoding="utf-8"
            ) as f:
                flat = [el for row in rows for el in row]
                json.dump(flat, f, indent=2, ensure_ascii=False)

            self.debug_draw_ocr(
                image,
                save_path=os.path.join(self.DUMP_PATH, "ocr_visual.png"),
            )

        chars  = len(result)
        tokens = chars // 4
        print(
            f"[parser] Characters: {chars} | Estimated tokens: {tokens}"
            f" | Estimated cost (Haiku): ${tokens * 0.000001:.6f}"
        )

        return result