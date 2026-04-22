from src.nanocontrol.vision.screen import *
from src.nanocontrol.vision.parser import *
import src.nanocontrol.utils as utils
import json
import time

timer = utils.timer()

timer.start()
img = screen_capture().capture_screen()
# ui_data = parser_engine(debug=False).get_screen_string_map(img)

# print(json.dumps(ui_data, indent=2))
# print(f"[*] Time to parse screen: {round(timer.stop(), 2)}s")

time_list = []

for i in range(10):
    t = time.perf_counter()
    parser_engine(debug=False).get_screen_string_map(img)
    time_list.append(f"Run {i}: {(time.perf_counter()-t)*1000:.0f}ms")

for i in range(len(time_list)):
    print(time_list[i])