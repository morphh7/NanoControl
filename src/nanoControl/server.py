"""
NanoControl WebSocket server.

Claude connects to ws://localhost:8765 and exchanges JSON messages.

Supported commands
------------------
{"action": "screenshot"}
    Capture the screen, run OCR, return the UI element map.
    Response: {"status": "ok", "elements": [...]}

{"action": "click", "x": 500, "y": 300}
{"action": "double_click", "x": 500, "y": 300}
{"action": "right_click", "x": 500, "y": 300}
{"action": "move_to", "x": 500, "y": 300}
    Mouse actions.
    Response: {"status": "ok"}

{"action": "scroll", "x": 500, "y": 300, "direction": "down", "amount": 3}
    Scroll at position. direction: "up" | "down", amount defaults to 3.
    Response: {"status": "ok"}

{"action": "drag", "x1": 100, "y1": 200, "x2": 400, "y2": 200}
    Click-drag from (x1,y1) to (x2,y2).
    Response: {"status": "ok"}

{"action": "type", "text": "hello world"}
    Type text (ASCII-safe). For unicode use action "paste".
    Response: {"status": "ok"}

{"action": "paste", "text": "héllo wörld"}
    Copy text to clipboard then Ctrl+V (handles unicode).
    Response: {"status": "ok"}

{"action": "press", "key": "enter"}
    Press a single key by name.
    Response: {"status": "ok"}

{"action": "hotkey", "keys": ["ctrl", "c"]}
    Send a key combination.
    Response: {"status": "ok"}

All errors return: {"status": "error", "message": "..."}
"""

import asyncio
import json
import traceback

import websockets

from nanoControl.vision.screen import capture_screen
from nanoControl.vision.parser import get_ui_map
from nanoControl.actions import mouse, keyboard

HOST = "localhost"
PORT = 8765


async def _handle(websocket):
    async for raw in websocket:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            await websocket.send(json.dumps({"status": "error", "message": f"invalid JSON: {e}"}))
            continue

        try:
            response = _dispatch(msg)
        except Exception:
            tb = traceback.format_exc()
            await websocket.send(json.dumps({"status": "error", "message": tb}))
            continue

        await websocket.send(json.dumps(response))


def _dispatch(msg: dict) -> dict:
    action = msg.get("action")

    if action == "screenshot":
        image = capture_screen()
        elements = get_ui_map(image)
        return {"status": "ok", "elements": elements}

    elif action == "click":
        mouse.click(msg["x"], msg["y"])

    elif action == "double_click":
        mouse.double_click(msg["x"], msg["y"])

    elif action == "right_click":
        mouse.right_click(msg["x"], msg["y"])

    elif action == "move_to":
        mouse.move_to(msg["x"], msg["y"])

    elif action == "scroll":
        mouse.scroll(msg["x"], msg["y"], msg["direction"], msg.get("amount", 3))

    elif action == "drag":
        mouse.drag(msg["x1"], msg["y1"], msg["x2"], msg["y2"])

    elif action == "type":
        keyboard.type_text(msg["text"])

    elif action == "paste":
        keyboard.type_text_raw(msg["text"])

    elif action == "press":
        keyboard.press_key(msg["key"])

    elif action == "hotkey":
        keyboard.hotkey(*msg["keys"])

    else:
        return {"status": "error", "message": f"unknown action: {action!r}"}

    return {"status": "ok"}


async def start_server():
    print(f"NanoControl listening on ws://{HOST}:{PORT}")
    async with websockets.serve(_handle, HOST, PORT):
        await asyncio.get_running_loop().create_future()  # run forever
