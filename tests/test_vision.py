import time
from PIL import Image
import json

from src import utils
from src.vision.parser import parser_engine
from src.vision.screen import screen_capture

def run_debug(engine: object, img: Image, loop_times: int = 10) -> None:
    time_list: list[str] = []
    
    for i in range(loop_times):
        t = time.perf_counter()
        engine.get_raw_screen_data(img)
        time_list.append(f"Run {i + 1}: {(time.perf_counter() - t) * 1000:.0f}ms")

    for line in time_list:
        print(line)

def main(debug: bool = False, loop: bool = False) -> None:
    timer = utils.timer()

    timer.start()
    img = screen_capture().capture_screen()
    print(f"[*] Screen captured in {timer.stop() * 1000:.0f}ms")

    engine = parser_engine(debug=debug)

    if loop: 
        run_debug(engine, img)
        return

    parsed_raw_data = engine.get_raw_screen_data(img)
    print(json.dumps(parsed_raw_data, indent=1))

if __name__ == "__main__":
    main(debug=True, loop=False)
