import os
from dotenv import load_dotenv
load_dotenv()

import anthropic
from flask import Flask, request, jsonify
from flask_cors import CORS

from src import utils
from src.vision.parser import parser_engine
from src.vision.screen import screen_capture

import pyautogui
import time

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── system prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are NanoControl, an autonomous PC agent.
Control the user's PC using tools. Rules:
- Call get_screen ONCE at the start only
- Do NOT call get_screen after every action
- Only call get_screen again if an action clearly failed
- Coordinates are compressed (multiply by 10 for real pixels)
- For simple tasks just execute directly without verifying each step
- Say done when complete"""

TOOLS = [
    {
        "name": "get_screen",
        "description": "Get parsed screen state",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "click",
        "description": "Click at coords",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"]
        }
    },
    {
        "name": "type_text",
        "description": "Type text",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]
        }
    },
    {
        "name": "press_key",
        "description": "Press key",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"]
        }
    }
]

# ── tool executor (this is where your parser + pyautogui will hook in) ──
engine = parser_engine(debug=False)
def execute_tool(name: str, inputs: dict) -> str:
    if name == "get_screen":
        img = screen_capture().capture_screen()
        return engine.get_efficient_screen_handle(img)

    elif name == "click":
        x, y = inputs["x"] * 10, inputs["y"] * 10
        pyautogui.click(x, y)
        time.sleep(0.5)  # wait for UI to respond
        return f"clicked at ({x}, {y})"

    elif name == "type_text":
        text = inputs["text"]
        pyautogui.typewrite(text, interval=0.05)
        return f"typed: {text}"

    elif name == "press_key":
        key = inputs["key"]
        pyautogui.press(key)
        return f"pressed: {key}"

    return f"unknown tool: {name}"

# ── main chat endpoint ─────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    all_messages = data.get('messages', [])
    messages = [all_messages[-1]]  # only latest user message

    total_input_tokens  = 0
    total_output_tokens = 0
    max_iterations = 10  # hard cap — prevents runaway loops

    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        total_input_tokens  += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        print(f"[loop {iterations}] in={response.usage.input_tokens} out={response.usage.output_tokens} stop={response.stop_reason}")

        text_blocks = [b.text for b in response.content if hasattr(b, 'text')]

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"[tool] {block.name} {block.input}")
                    result = execute_tool(block.name, block.input)
                    print(f"[tool result] {result[:100]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})

        else:
            final_text = "\n".join(text_blocks) or "done"
            print(f"[done] {iterations} iterations, {total_input_tokens} in, {total_output_tokens} out")
            return jsonify({
                "content": [{"type": "text", "text": final_text}],
                "usage": {
                    "input_tokens":  total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "tool_calls":    iterations,
                }
            })

    # hit the cap
    return jsonify({
        "content": [{"type": "text", "text": f"Stopped after {max_iterations} steps. Task may be incomplete."}],
        "usage": {
            "input_tokens":  total_input_tokens,
            "output_tokens": total_output_tokens,
            "tool_calls":    max_iterations,
        }
    })

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)