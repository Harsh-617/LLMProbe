# Root app.py — entry point for HuggingFace Spaces
# This just imports and launches the dashboard

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from dashboard.app import build_app

app = build_app()
app.launch()