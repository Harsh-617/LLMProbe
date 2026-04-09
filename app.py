import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dashboard.app import build_app

app = build_app()
app.launch(server_name="0.0.0.0", server_port=7860)