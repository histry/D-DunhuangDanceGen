"""Keep legacy Native policy variables, isolate each worker's module namespace."""
import os
from pathlib import Path
def worker_environment():
    env=dict(os.environ)
    env["PYTHONPATH"]=str(Path(__file__).resolve().parents[3])
    return env
