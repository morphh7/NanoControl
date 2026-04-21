import pyautogui
import os
from PIL import Image

def capture_screen() -> Image.Image:
    """
    Capture a screenshot of the screen.
    Also saves ss to /temp for debugging

    - param: none
    - return: PIL image
    """

    temp_dump_dir = "C:/Users/bilguun.odbayar/Documents/GitHub/NanoControl/src/nanocontrol/temp"
    screenshot = pyautogui.screenshot(temp_dump_dir + "/temp_ss.png")
    
    return screenshot