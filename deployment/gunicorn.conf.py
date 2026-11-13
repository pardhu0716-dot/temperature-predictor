"""Production settings without changes to original Stage A files."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
chdir = str(REPO)
pythonpath = ','.join([str(REPO / 'temperature_project'), str(REPO / 'deployment')])
bind = f'0.0.0.0:{int(os.environ.get("PORT", "10000"))}'
workers = 1
worker_class = 'gthread'
threads = 4
preload_app = True
timeout = 60
graceful_timeout = 30
keepalive = 5
max_requests = 2000
max_requests_jitter = 100
accesslog = '-'
errorlog = '-'
capture_output = True
loglevel = 'info'
control_socket_disable = True
