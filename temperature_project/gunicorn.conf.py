"""Small Render Free instance: one process and four request threads."""
import os
from pathlib import Path

chdir = str(Path(__file__).resolve().parent)
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
# No management socket is needed for this read-only Render web service.
control_socket_disable = True
