#!/usr/bin/env python3
"""
Local combined Stage A + Stage B + Stage C Flask entry point.

Self-contained additive wrapper:
  - Imports the existing Flask app from combined_app.py (Stage A '/' + Stage B '/stage-b').
  - Registers the isolated Stage C Blueprint ('/stage-c' and '/api/stage-c/*')
    from stage_c/web/ inside this repository.
  - Leaves app.py, combined_app.py, Stage A, and Stage B completely untouched.

Local startup (port 5205):
    cd temperature_project
    python -m flask --app combined_abc_app run --host 127.0.0.1 --port 5205
"""

from __future__ import annotations

import os

from combined_app import app  # Registers Stage A ('/') and Stage B ('/stage-b')
from stage_c.web import register_stage_c

register_stage_c(app)


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5205"))
    app.run(host=host, port=port, debug=False)
