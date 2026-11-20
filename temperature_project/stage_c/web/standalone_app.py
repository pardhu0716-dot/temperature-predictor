#!/usr/bin/env python3
"""
Standalone Stage C Flask Application for Isolated Local Testing (Port 5051).

Runs Stage C completely independently of Stage A and Stage B.
Usage:
    cd temperature_project
    python -m stage_c.web.standalone_app
"""

from __future__ import annotations

import os

from flask import Flask, redirect

from .blueprint import register_stage_c


def create_standalone_app() -> Flask:
    app = Flask(__name__)
    register_stage_c(app)

    @app.get("/")
    def _standalone_root_redirect():
        return redirect("/stage-c", code=302)

    return app


app = create_standalone_app()


if __name__ == "__main__":
    port = int(os.environ.get("STAGE_C_PORT", "5051"))
    host = os.environ.get("STAGE_C_HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False)
