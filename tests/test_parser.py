import time
from PIL import Image
import json

from src import utils
from src.vision.parser import parser_engine
from src.vision.screen import screen_capture

def main(debug: bool = False) -> None:
    timer = utils.timer()

    timer.start()
    img = screen_capture().capture_screen()
    
    print(f"[*] Screen captured in {timer.stop() * 1000:.0f}ms")

    engine = parser_engine(debug=debug)

    final_result = engine.get_efficient_screen_handle(img)

    chars = len(final_result)
    approx_tokens = chars // 4

    print(final_result)
    print(f"[*] Characters: {chars}")
    print(f"[*] Estimated tokens: {approx_tokens}")
    print(f"[*] Estimated cost (Haiku): ${approx_tokens * 0.000001:.6f}")

if __name__ == "__main__":
    main(debug=True)