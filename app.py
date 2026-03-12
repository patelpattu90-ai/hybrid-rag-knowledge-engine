# app.py — HuggingFace Spaces entry point
# HF Spaces looks for app.py in the root directory.
# This simply launches the Gradio app from app/ui.py.

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ui import demo

if __name__ == "__main__":
    demo.launch()
