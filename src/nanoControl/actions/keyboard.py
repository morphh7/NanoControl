import pyautogui


def type_text(text: str, interval: float = 0.02) -> None:
    """Type a string character by character."""
    pyautogui.typewrite(text, interval=interval)


def type_text_raw(text: str) -> None:
    """Type text including unicode via clipboard paste (handles special chars)."""
    import pyperclip
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")


def press_key(key: str) -> None:
    """Press a single key (e.g. 'enter', 'tab', 'escape', 'f5')."""
    pyautogui.press(key)


def hotkey(*keys: str) -> None:
    """Send a key combination (e.g. hotkey('ctrl', 'c'))."""
    pyautogui.hotkey(*keys)
