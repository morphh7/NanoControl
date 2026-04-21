import pyautogui
from PIL import Image


def capture_screen(save_path: str | None = None) -> Image.Image:
    """
    Capture a full screenshot.

    - param save_path: optional file path to persist the image for debugging
    - return: PIL image
    """
    screenshot = pyautogui.screenshot()

    if save_path:
        screenshot.save(save_path)

    return screenshot
