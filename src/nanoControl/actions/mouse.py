import pyautogui

# Prevent pyautogui from throwing on fast moves; keep a small pause for stability
pyautogui.PAUSE = 0.05
pyautogui.FAILSAFE = True  # move mouse to top-left corner to abort


def click(x: int, y: int) -> None:
    pyautogui.click(x, y)


def double_click(x: int, y: int) -> None:
    pyautogui.doubleClick(x, y)


def right_click(x: int, y: int) -> None:
    pyautogui.rightClick(x, y)


def move_to(x: int, y: int, duration: float = 0.15) -> None:
    pyautogui.moveTo(x, y, duration=duration)


def scroll(x: int, y: int, direction: str, amount: int = 3) -> None:
    """direction: 'up' or 'down'"""
    pyautogui.moveTo(x, y)
    clicks = amount if direction == "up" else -amount
    pyautogui.scroll(clicks)


def drag(x1: int, y1: int, x2: int, y2: int, duration: float = 0.3) -> None:
    pyautogui.moveTo(x1, y1)
    pyautogui.dragTo(x2, y2, duration=duration, button="left")
