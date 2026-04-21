from paddleocr import PaddleOCR
import numpy as np
import json

# init OCR
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

def get_ui_map(image: any) -> list[dict]:
    """
    Map out elements from taken screenshot a list of text 
    elements with cordiantes for the respective element

    - param: image (screenshot)
    - return: element and screen positions
    """
    
    img_array = np.array(image)
    ocr_result = ocr_engine.ocr(img_array)
    ui_elements = []

    temp_dump_dir = "C:/Users/bilguun.odbayar/Documents/GitHub/NanoControl/src/nanocontrol/temp"
    with open(temp_dump_dir + "/recent_ocr_dump.txt", "a") as f:
        f.write(str(ocr_result))

    if ocr_result and ocr_result[0]:
        for detection in ocr_result[0]:
            corners = detection[0] # detected within this "box"

            text_info = detection[1] 
            text = text_info[0] # strip the actual text content
            confidence = text_info[1] # detection confidence

            center_x = int(sum(p[0] for p in corners) / 4)
            center_y = int(sum(p[1] for p in corners) / 4)

            ui_elements.append({
                "text": text,
                "pos": [center_x, center_y],
                "conf": round(float(confidence), 2)
            })

    with open(temp_dump_dir + "/recent_ocr_dump.json", "a") as f:
        f.write(json.dumps(ui_elements, indent=2))

    return ui_elements