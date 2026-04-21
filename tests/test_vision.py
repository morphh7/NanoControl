from nanocontrol.vision.screen import capture_screen
from nanocontrol.vision.parser import get_ui_map
import json

img = capture_screen()
ui_data = get_ui_map(img)

print(json.dumps(ui_data, indent=2)) 