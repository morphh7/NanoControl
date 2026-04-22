import pyautogui
import os
from PIL import Image
import src.nanocontrol.utils as utils

class screen_capture:
    def __init__(self):
        self.PATH = utils.path()

    def capture_screen(self) -> Image.Image:
        """
        Capture a screenshot of the screen.
        Also saves ss to /temp for debugging

        - param: none
        - return: PIL image
        """

        temp_dump_dir = self.PATH.find_and_create_temp()
        screenshot = pyautogui.screenshot(temp_dump_dir + "/temp_ss.png")
    
        return screenshot