from setuptools import setup, find_packages

setup(
    name="nanocontrol",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=[
        "pyautogui",
        "Pillow",
        "numpy<2.0",
        "paddlepaddle==2.6.2",
        "paddleocr",
        "websockets",
        "pyperclip",
    ],
)
