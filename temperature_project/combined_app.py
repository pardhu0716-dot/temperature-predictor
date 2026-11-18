"""Local combined Stage A + Stage B entry point. Original app.py remains unchanged."""
from app import app
from stage_b.web import register_stage_b

register_stage_b(app)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
